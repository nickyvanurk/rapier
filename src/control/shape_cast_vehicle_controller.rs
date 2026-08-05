//! A vehicle controller based on cylinder shape-casting, ported and modified from Bullet’s `btRaycastVehicle`.

use crate::dynamics::{RigidBody, RigidBodyHandle, RigidBodySet};
use crate::geometry::{ColliderHandle, ColliderSet, Cylinder};
use crate::math::{Isometry, Point, Real, Rotation, Translation, Vector};
use crate::pipeline::QueryPipeline;
use crate::prelude::QueryPipelineMut;
use crate::utils::{SimdCross, SimdDot};
use na::{DMatrix, DVector};
use parry::query::details::ShapeCastOptions;

/// A character controller to simulate vehicles using cylinder shape-casting for the wheels.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct DynamicShapeCastVehicleController {
    wheels: Vec<Wheel>,
    forward_ws: Vec<Vector<Real>>,
    axle: Vec<Vector<Real>>,
    /// The current forward speed of the vehicle.
    pub current_vehicle_speed: Real,

    /// Handle of the vehicle’s chassis.
    pub chassis: RigidBodyHandle,
    /// The chassis’ local _up_ direction (`0 = x, 1 = y, 2 = z`)
    pub index_up_axis: usize,
    /// The chassis’ local _forward_ direction (`0 = x, 1 = y, 2 = z`)
    pub index_forward_axis: usize,
}

#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq)]
/// Parameters affecting the physical behavior of a wheel.
pub struct WheelTuning {
    /// The suspension stiffness.
    ///
    /// Increase this value if the suspension appears to not push the vehicle strong enough.
    pub suspension_stiffness: Real,
    /// The suspension’s damping when it is being compressed.
    pub suspension_compression: Real,
    /// The suspension’s damping when it is being released.
    ///
    /// Increase this value if the suspension appears to overshoot.
    pub suspension_damping: Real,
    /// The maximum distance the suspension can travel before and after its resting length.
    pub max_suspension_travel: Real,
    /// The multiplier of friction between a tire and the collider it's on top of.
    pub side_friction_stiffness: Real,
    /// Parameter controlling how much traction the tire has.
    ///
    /// The larger the value, the more instantaneous braking will happen (with the risk of
    /// causing the vehicle to flip if it’s too strong).
    pub friction_slip: Real,
    /// The maximum force applied by the suspension.
    pub max_suspension_force: Real,
}

impl Default for WheelTuning {
    fn default() -> Self {
        Self {
            suspension_stiffness: 5.88,
            suspension_compression: 0.83,
            suspension_damping: 0.88,
            max_suspension_travel: 5.0,
            side_friction_stiffness: 1.0,
            friction_slip: 10.5,
            max_suspension_force: 6000.0,
        }
    }
}

/// Objects used to initialize a wheel.
struct WheelDesc {
    /// The position of the wheel, relative to the chassis.
    pub chassis_connection_cs: Point<Real>,
    /// The direction of the wheel’s suspension, relative to the chassis.
    ///
    /// The shape-casting will happen following this direction to detect the ground.
    pub direction_cs: Vector<Real>,
    /// The wheel’s axle axis, relative to the chassis.
    pub axle_cs: Vector<Real>,
    /// The rest length of the wheel’s suspension spring.
    pub suspension_rest_length: Real,
    /// The maximum distance the suspension can travel before and after its resting length.
    pub max_suspension_travel: Real,
    /// The wheel’s radius.
    pub radius: Real,
    /// The wheel’s width (cylinder length along the axle).
    pub width: Real,

    /// The suspension stiffness.
    ///
    /// Increase this value if the suspension appears to not push the vehicle strong enough.
    pub suspension_stiffness: Real,
    /// The suspension’s damping when it is being compressed.
    pub damping_compression: Real,
    /// The suspension’s damping when it is being released.
    ///
    /// Increase this value if the suspension appears to overshoot.
    pub damping_relaxation: Real,
    /// Parameter controlling how much traction the tire has.
    ///
    /// The larger the value, the more instantaneous braking will happen (with the risk of
    /// causing the vehicle to flip if it’s too strong).
    pub friction_slip: Real,
    /// The maximum force applied by the suspension.
    pub max_suspension_force: Real,
    /// The multiplier of friction between a tire and the collider it's on top of.
    pub side_friction_stiffness: Real,
}

#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq)]
/// A wheel attached to a vehicle.
pub struct Wheel {
    shape_cast_info: ShapeCastInfo,

    center: Point<Real>,
    wheel_direction_ws: Vector<Real>,
    wheel_axle_ws: Vector<Real>,

    /// The position of the wheel, relative to the chassis.
    pub chassis_connection_point_cs: Point<Real>,
    /// The direction of the wheel’s suspension, relative to the chassis.
    ///
    /// The shape-casting will happen following this direction to detect the ground.
    pub direction_cs: Vector<Real>,
    /// The wheel’s axle axis, relative to the chassis.
    pub axle_cs: Vector<Real>,
    /// The rest length of the wheel’s suspension spring.
    pub suspension_rest_length: Real,
    /// The maximum distance the suspension can travel before and after its resting length.
    pub max_suspension_travel: Real,
    /// The wheel’s radius.
    pub radius: Real,
    /// The wheel’s width (cylinder length along the axle).
    pub width: Real,
    /// The suspension stiffness.
    ///
    /// Increase this value if the suspension appears to not push the vehicle strong enough.
    pub suspension_stiffness: Real,
    /// The suspension’s damping when it is being compressed.
    pub damping_compression: Real,
    /// The suspension’s damping when it is being released.
    ///
    /// Increase this value if the suspension appears to overshoot.
    pub damping_relaxation: Real,
    /// Parameter controlling how much traction the tire has.
    ///
    /// The larger the value, the more instantaneous braking will happen (with the risk of
    /// causing the vehicle to flip if it’s too strong).
    pub friction_slip: Real,
    /// The multiplier of friction between a tire and the collider it's on top of.
    pub side_friction_stiffness: Real,
    /// The wheel’s current rotation on its axle.
    pub rotation: Real,
    delta_rotation: Real,
    roll_influence: Real, // TODO: make this public?
    /// Anti-squat factor in `[0, 1]`: vertical fraction from contact point to
    /// chassis center-of-mass at which longitudinal (engine/brake) impulse
    /// is applied. `0` = applied at contact (Bullet default, full pitch
    /// moment). `1` = applied at COM height (zero pitch moment).
    pub anti_squat: Real,
    /// Reject a shape-cast hit when the wheel's suspension direction
    /// points more than this angle away from world-down. In radians,
    /// default 80°. Rejects hits when the chassis is sideways,
    /// on its nose/tail, or fully inverted — in those poses the cast
    /// would otherwise find ground the wrong way round and the spring
    /// would push the chassis deeper into terrain. The hit normal is
    /// not a useful signal here: parry returns the static shape's
    /// outward normal (world-up for a heightfield) regardless of which
    /// side the cast came from.
    pub suspension_reject_angle: Real,
    /// Reject a shape-cast hit when the contact-surface normal points
    /// more than this angle away from the anti-suspension direction.
    /// In radians, default 80° — same boundary as
    /// `suspension_reject_angle` so the two filters reject symmetric
    /// cases. Stops the wheel cylinder from "climbing" walls it clips
    /// when the chassis is pressed against them: without this the cast
    /// returns a `toi=0` hit and the bump-stop teleports the chassis
    /// upward.
    pub surface_reject_angle: Real,
    /// The maximum force applied by the suspension.
    pub max_suspension_force: Real,

    /// The forward impulses applied by the wheel on the chassis.
    pub forward_impulse: Real,
    /// The side impulses applied by the wheel on the chassis.
    pub side_impulse: Real,

    /// The steering angle for this wheel.
    pub steering: Real,
    /// The forward force applied by this wheel on the chassis.
    pub engine_force: Real,
    /// The maximum amount of braking impulse applied to slow down the vehicle.
    pub brake: Real,

    /// World-space point this wheel's contact is stuck to while static
    /// friction holds, or `None` when the wheel is rolling freely.
    ///
    /// Velocity-level friction can only cancel the velocity that exists when
    /// it runs, so a persistent horizontal force re-accelerates the chassis
    /// every step and the `v * dt` travelled before the cancel is never
    /// repaid — the vehicle creeps with its velocity pinned near zero. Real
    /// static friction holds a contact *point*, so the anchor records where
    /// the contact stuck and the drift away from it is driven back out.
    static_friction_anchor: Option<Point<Real>>,

    clipped_inv_contact_dot_suspension: Real,
    suspension_relative_velocity: Real,
    /// The force applied by the suspension.
    pub wheel_suspension_force: Real,
    skid_info: Real,
}

impl Wheel {
    fn new(info: WheelDesc) -> Self {
        Self {
            shape_cast_info: ShapeCastInfo::default(),
            suspension_rest_length: info.suspension_rest_length,
            max_suspension_travel: info.max_suspension_travel,
            radius: info.radius,
            width: info.width,
            suspension_stiffness: info.suspension_stiffness,
            damping_compression: info.damping_compression,
            damping_relaxation: info.damping_relaxation,
            chassis_connection_point_cs: info.chassis_connection_cs,
            direction_cs: info.direction_cs,
            axle_cs: info.axle_cs,
            wheel_direction_ws: info.direction_cs,
            wheel_axle_ws: info.axle_cs,
            center: Point::origin(),
            friction_slip: info.friction_slip,
            steering: 0.0,
            engine_force: 0.0,
            rotation: 0.0,
            delta_rotation: 0.0,
            brake: 0.0,
            roll_influence: 0.1,
            anti_squat: 0.5,
            // 80° from upright. Accepts chassis tilt up to 80° (plenty
            // of margin above a typical 45° drivable-slope ceiling),
            // rejects sideways, nose/tail-standing, and inverted poses.
            // A tighter π/2 boundary is floating-point sensitive —
            // near-90° tilts slipped through.
            suspension_reject_angle: 80.0 * std::f32::consts::PI / 180.0,
            // 80° from the wheel's anti-suspension direction — same as
            // suspension_reject_angle. Accepts slopes up to 80°, rejects
            // walls (90°) and ceilings the cast cylinder clips when the
            // chassis is pressed against them.
            surface_reject_angle: 80.0 * std::f32::consts::PI / 180.0,
            static_friction_anchor: None,
            clipped_inv_contact_dot_suspension: 0.0,
            suspension_relative_velocity: 0.0,
            wheel_suspension_force: 0.0,
            max_suspension_force: info.max_suspension_force,
            skid_info: 0.0,
            side_impulse: 0.0,
            forward_impulse: 0.0,
            side_friction_stiffness: info.side_friction_stiffness,
        }
    }

    /// Information about suspension and the ground obtained from the shape-casting
    /// for this wheel.
    pub fn shape_cast_info(&self) -> &ShapeCastInfo {
        &self.shape_cast_info
    }

    /// The world-space center of the wheel.
    pub fn center(&self) -> Point<Real> {
        self.center
    }

    /// The world-space direction of the wheel’s suspension.
    pub fn suspension(&self) -> Vector<Real> {
        self.wheel_direction_ws
    }

    /// The world-space direction of the wheel’s axle.
    pub fn axle(&self) -> Vector<Real> {
        self.wheel_axle_ws
    }
}

/// Information about suspension and the ground obtained from the shape-casting
/// to simulate a wheel’s suspension.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct ShapeCastInfo {
    /// World-space suspension-force normal. Follows the shape-cast ground
    /// normal (Bullet's original convention), with an oblique fallback to the
    /// suspension axis (`-wheel_direction_ws`) when a wheel clips a vertical
    /// face so the spring can't push the chassis sideways. Using the real
    /// ground normal re-enables the `clipped_inv_contact_dot_suspension`
    /// slope correction; coincides with `ground_normal_ws` except in the
    /// no-contact branch.
    pub contact_normal_ws: Vector<Real>,
    /// World-space ground-surface normal from the shape-cast hit. Used to
    /// build `forward_ws`/`axle_ws` so drive/brake friction stays in the
    /// real contact plane (otherwise a pitched chassis gains a vertical
    /// component in its drive force and lifts off under throttle).
    pub ground_normal_ws: Vector<Real>,
    /// The (world-space) point hit by the wheel’s shape-cast.
    pub contact_point_ws: Point<Real>,
    /// The suspension length for the wheel.
    pub suspension_length: Real,
    /// The (world-space) starting point of the shape-cast (the chassis-side suspension anchor).
    pub hard_point_ws: Point<Real>,
    /// Is the wheel in contact with the ground?
    pub is_in_contact: bool,
    /// The collider hit by the shape-cast.
    pub ground_object: Option<ColliderHandle>,
    /// True when the shape-cast would have placed the wheel past the hard
    /// point (suspension fully compressed). The spring is effectively a
    /// rigid stop; callers should apply a plastic contact impulse instead
    /// of relying on the spring alone.
    pub bottomed_out: bool,
    /// Signed overshoot in world units along the suspension direction when
    /// `bottomed_out` is true — how far the chassis is penetrating past the
    /// hard-point contact. Used for position correction; zero otherwise.
    pub bottom_out_overshoot: Real,
    /// Debug: raw TOI returned by the last shape-cast hit.
    pub last_toi: Real,
    /// Debug: starting-position offset (along `-direction`) used for the last shape-cast.
    pub last_cast_offset: Real,
    /// Debug: witness point on the cylinder side (local to cylinder, world space).
    pub last_witness2: Point<Real>,
    /// Debug: cylinder starting center (world space).
    pub last_cast_start: Point<Real>,
    /// Debug: world-space axle used for the last shape-cast.
    pub last_axle_ws: Vector<Real>,
}

impl DynamicShapeCastVehicleController {
    /// Creates a new vehicle represented by the given rigid-body.
    ///
    /// Wheels have to be attached afterwards calling [`Self::add_wheel`].
    pub fn new(chassis: RigidBodyHandle) -> Self {
        Self {
            wheels: vec![],
            forward_ws: vec![],
            axle: vec![],
            current_vehicle_speed: 0.0,
            chassis,
            index_up_axis: 1,
            index_forward_axis: 0,
        }
    }

    //
    // basically most of the code is general for 2 or 4 wheel vehicles, but some of it needs to be reviewed
    //
    /// Adds a wheel to this vehicle.
    pub fn add_wheel(
        &mut self,
        chassis_connection_cs: Point<Real>,
        direction_cs: Vector<Real>,
        axle_cs: Vector<Real>,
        suspension_rest_length: Real,
        radius: Real,
        width: Real,
        tuning: &WheelTuning,
    ) -> &mut Wheel {
        let ci = WheelDesc {
            chassis_connection_cs,
            direction_cs,
            axle_cs,
            suspension_rest_length,
            radius,
            width,
            suspension_stiffness: tuning.suspension_stiffness,
            damping_compression: tuning.suspension_compression,
            damping_relaxation: tuning.suspension_damping,
            friction_slip: tuning.friction_slip,
            max_suspension_travel: tuning.max_suspension_travel,
            max_suspension_force: tuning.max_suspension_force,
            side_friction_stiffness: tuning.side_friction_stiffness,
        };

        let wheel_id = self.wheels.len();
        self.wheels.push(Wheel::new(ci));

        &mut self.wheels[wheel_id]
    }

    fn update_wheel_transform(&mut self, chassis: &RigidBody, wheel_index: usize) {
        self.update_wheel_transforms_ws(chassis, wheel_index);
        let wheel = &mut self.wheels[wheel_index];

        let steering_orn = Rotation::new(-wheel.wheel_direction_ws * wheel.steering);
        wheel.wheel_axle_ws = steering_orn * (chassis.position() * wheel.axle_cs);
        wheel.center = wheel.shape_cast_info.hard_point_ws
            + wheel.wheel_direction_ws * wheel.shape_cast_info.suspension_length;
    }

    fn update_wheel_transforms_ws(&mut self, chassis: &RigidBody, wheel_id: usize) {
        let wheel = &mut self.wheels[wheel_id];
        wheel.shape_cast_info.is_in_contact = false;

        let chassis_transform = chassis.position();

        wheel.shape_cast_info.hard_point_ws = chassis_transform * wheel.chassis_connection_point_cs;
        wheel.wheel_direction_ws = chassis_transform * wheel.direction_cs;
        wheel.wheel_axle_ws = chassis_transform * wheel.axle_cs;
    }

    #[profiling::function]
    fn shape_cast(
        &mut self,
        queries: &QueryPipeline,
        chassis: &RigidBody,
        wheel_id: usize,
        dt: Real,
    ) {
        let world_up = Vector::ith(self.index_up_axis, 1.0);
        let wheel = &mut self.wheels[wheel_id];
        let source = wheel.shape_cast_info.hard_point_ws;

        // Parry’s `Cylinder` has its axis along +Y. Align that axis with the wheel axle.
        let axle = wheel
            .wheel_axle_ws
            .try_normalize(1.0e-5)
            .unwrap_or_else(|| Vector::y());
        let cyl_rot = Rotation::rotation_between(&Vector::y(), &axle)
            .unwrap_or_else(Rotation::identity);
        let cylinder = Cylinder::new(wheel.width * 0.5, wheel.radius);

        let direction = wheel
            .wheel_direction_ws
            .try_normalize(1.0e-5)
            .unwrap_or(wheel.wheel_direction_ws);

        // Start the cylinder one radius behind the hard_point (opposite
        // to cast direction) so its near cap sits at the hard_point. The
        // smallest offset that keeps the cylinder from starting inside
        // the wheel-at-rest position; minimises overlap with external
        // geometry above the wheel mount (low ceilings, characters
        // walking next to the chassis). Detection range stays the same
        // because `max_time_of_impact = raylen + offset` grows with
        // `offset`. suspension_length = toi − offset, matching the
        // raycast's hit_distance − radius formula.
        let r = wheel.radius;
        let raylen = wheel.suspension_rest_length + r;
        let offset = r;

        let hit = {
            let pos = Isometry::from_parts(
                Translation::from(source.coords - direction * offset),
                cyl_rot,
            );
            let options = ShapeCastOptions {
                max_time_of_impact: raylen + offset,
                target_distance: 0.0,
                // Skip "starts in penetration" hits when the cast
                // trajectory exits the penetration. Without this, anything
                // overlapping the cast cylinder above the wheel mount (low
                // ceiling, character walking next to the chassis) reports
                // toi=0 and the bump-stop teleports the chassis upward by
                // `offset` per tick. Genuine bottom-out (cylinder inside
                // the ground) still reports because moving down doesn't
                // exit the ground penetration.
                stop_at_penetration: false,
                compute_impact_geometry_on_penetration: true,
            };
            queries.cast_shape(&pos, &direction, &cylinder, options)
        };

        // Reject hits in two complementary ways. Both filters target
        // the same failure mode (the spring pushing the chassis the
        // wrong way), but they catch different geometry:
        //
        // 1. `suspension_reject_angle` — chassis-pose check. Drops the
        //    hit when the chassis is sideways/inverted, where parry's
        //    heightfield convention (always +Y outward) would mislead a
        //    pure normal-based check.
        // 2. `surface_reject_angle` — hit-normal check. Drops the hit
        //    when the contact surface isn't drivable from the wheel
        //    side. The cast cylinder extends radially around its axle,
        //    so when the chassis is pressed against a wall the cylinder
        //    clips it and reports a `toi=0` hit; without this filter
        //    the bump-stop teleports the chassis up `offset` per tick
        //    ("climbs" the wall).
        let suspension_downward = -direction.dot(&world_up); // 1 when down, -1 when up
        let hit = if suspension_downward > wheel.suspension_reject_angle.cos() {
            hit
        } else {
            None
        };
        let hit = hit.filter(|(_, h)| {
            // Compare against world-up, not the suspension direction.
            // Drivability is a property of the surface in world space
            // (a 45° slope is always a 45° slope), independent of how
            // the chassis is currently oriented. Using suspension
            // direction would let chassis tilt shift the accept/reject
            // boundary unpredictably.
            let world_up_facing = h.normal1.into_inner().dot(&world_up);
            world_up_facing > wheel.surface_reject_angle.cos()
        });

        wheel.shape_cast_info.ground_object = None;

        if let Some((collider_hit, hit)) = hit {
            // Suspension and friction both use the real ground normal (Bullet's
            // original convention) so the spring reaction stays aligned with the
            // actual contact plane on slopes — an axial-locked spring leaves a
            // lateral residual on uneven terrain (creep/jitter). Fall back to
            // axial when the raw hit is > ~60° off axis (wheel wedged against a
            // wall isn't drivable ground) — this guard is what stops the spring
            // from shoving the chassis sideways off a vertical face.
            let axial = -wheel.wheel_direction_ws;
            let raw = hit.normal1.into_inner();
            let ground_normal = if raw.dot(&axial) > 0.5 { raw } else { axial };

            wheel.shape_cast_info.contact_normal_ws = ground_normal;
            wheel.shape_cast_info.ground_normal_ws = ground_normal;
            wheel.shape_cast_info.is_in_contact = true;
            wheel.shape_cast_info.ground_object = Some(collider_hit);

            let raw_length = hit.time_of_impact - offset;
            wheel.shape_cast_info.last_toi = hit.time_of_impact;
            wheel.shape_cast_info.last_cast_offset = offset;
            wheel.shape_cast_info.last_witness2 = hit.witness2;
            wheel.shape_cast_info.last_cast_start = Point::from(source.coords - direction * offset);
            wheel.shape_cast_info.last_axle_ws = axle;

            // The wheel can't travel past the hard point — the spring has
            // a mechanical end stop. A negative `raw_length` means the
            // cast would have put the wheel above the hard point: the
            // chassis is overshooting into the ground and the remaining
            // compression distance has to be absorbed rigidly, not by
            // spring force.
            let max_suspension_length = wheel.suspension_rest_length + wheel.max_suspension_travel;
            let target_length = raw_length.clamp(0.0, max_suspension_length);
            wheel.shape_cast_info.bottomed_out = raw_length < 0.0;
            wheel.shape_cast_info.bottom_out_overshoot = (-raw_length).max(0.0);
            wheel.shape_cast_info.suspension_length = target_length;
            wheel.shape_cast_info.contact_point_ws = hit.witness1;

            let denominator = wheel
                .shape_cast_info
                .contact_normal_ws
                .dot(&wheel.wheel_direction_ws);
            let chassis_velocity_at_contact_point =
                chassis.velocity_at_point(&wheel.shape_cast_info.contact_point_ws);
            let proj_vel = wheel
                .shape_cast_info
                .contact_normal_ws
                .dot(&chassis_velocity_at_contact_point);

            if denominator >= -0.1 {
                wheel.suspension_relative_velocity = 0.0;
                wheel.clipped_inv_contact_dot_suspension = 1.0 / 0.1;
            } else {
                let inv = -1.0 / denominator;
                wheel.suspension_relative_velocity = proj_vel * inv;
                wheel.clipped_inv_contact_dot_suspension = inv;
            }
        } else {
            // No contact. Droop toward the full extension stop
            // (`rest + max_travel`), not rest: a real suspension hangs at
            // its droop limit under gravity, and pre-extending the wheel
            // while airborne means there's little to catch up on when the
            // ground reappears — reduces the landing-side snap that
            // Bullet's btRaycastVehicle is known for. ~200 ms TC.
            let droop_length = wheel.suspension_rest_length + wheel.max_suspension_travel;
            let time_constant = 0.2;
            let factor = 1.0 - (-dt / time_constant).exp();
            wheel.shape_cast_info.suspension_length +=
                (droop_length - wheel.shape_cast_info.suspension_length) * factor;
            wheel.shape_cast_info.bottomed_out = false;
            wheel.shape_cast_info.bottom_out_overshoot = 0.0;
            wheel.suspension_relative_velocity = 0.0;
            wheel.shape_cast_info.contact_normal_ws = -wheel.wheel_direction_ws;
            wheel.shape_cast_info.ground_normal_ws = -wheel.wheel_direction_ws;
            wheel.clipped_inv_contact_dot_suspension = 1.0;
        }
    }

    /// Updates the vehicle’s velocity based on its suspension, engine force, and brake.
    #[profiling::function]
    pub fn update_vehicle(&mut self, dt: Real, queries: QueryPipelineMut) {
        let num_wheels = self.wheels.len();
        let chassis = &queries.bodies[self.chassis];

        for i in 0..num_wheels {
            self.update_wheel_transform(chassis, i);
        }

        self.current_vehicle_speed = chassis.linvel().norm();

        let forward_w = chassis.position() * Vector::ith(self.index_forward_axis, 1.0);

        if forward_w.dot(chassis.linvel()) < 0.0 {
            self.current_vehicle_speed *= -1.0;
        }

        //
        // simulate suspension
        //

        for wheel_id in 0..self.wheels.len() {
            self.shape_cast(&queries.as_ref(), chassis, wheel_id, dt);
        }

        let chassis_mass = chassis.mass();
        self.update_suspension(chassis_mass, dt);

        let chassis = queries
            .bodies
            .get_mut_internal_with_modification_tracking(self.chassis)
            .unwrap();

        for wheel in &mut self.wheels {
            if wheel.engine_force > 0.0 {
                chassis.wake_up(true);
            }

            // apply suspension force
            let mut suspension_force = wheel.wheel_suspension_force;

            if suspension_force > wheel.max_suspension_force {
                suspension_force = wheel.max_suspension_force;
            }

            let impulse = wheel.shape_cast_info.contact_normal_ws * suspension_force * dt;
            chassis.apply_impulse_at_point(impulse, wheel.shape_cast_info.contact_point_ws, false);
        }

        // Bump-stop velocity constraint — direct solve across all bottomed
        // wheels (Havok's recommended pattern for vehicle suspensions:
        // single-pass simultaneous solve beats iterative PGS when multiple
        // unilateral contacts share a body).
        //
        // For each bottomed wheel i with contact `p_i` and normal `n_i`,
        // we enforce `v_at(p_i) · n_i ≥ 0`. Write the velocity change
        // produced by impulses λ along each normal as:
        //     Δv_at(p_i) · n_i = Σ_j λ_j · A[i][j]
        // where A[i][j] = n_i · M⁻¹ · n_j + (r_i × n_i) · I⁻¹ · (r_j × n_j).
        // We want v + Δv · n ≥ 0, with λ ≥ 0 and complementary slackness
        // (a wheel with λ_i > 0 is active; its constraint is exactly met).
        //
        // Active-set: start with every bottomed+inward-moving wheel, solve
        // A·λ = b, drop any wheel where λ_i < 0 (it would have to pull the
        // chassis down), resolve. With N ≤ ~8 wheels this is a handful of
        // tiny dense solves.
        {
            #[derive(Clone, Copy)]
            struct BottomedWheel {
                contact: Point<Real>,
                normal: Vector<Real>,
                rel_vel: Real,
            }

            let mut candidates: Vec<BottomedWheel> = self
                .wheels
                .iter()
                .filter(|w| w.shape_cast_info.bottomed_out)
                .filter_map(|w| {
                    let normal = w.shape_cast_info.contact_normal_ws;
                    let contact = w.shape_cast_info.contact_point_ws;
                    let vel = chassis.velocity_at_point(&contact);
                    let rel_vel = normal.dot(&vel);
                    (rel_vel < 0.0).then_some(BottomedWheel {
                        contact,
                        normal,
                        rel_vel,
                    })
                })
                .collect();

            if !candidates.is_empty() {
                let com = chassis.center_of_mass();
                let inv_mass = chassis.mprops.local_mprops.inv_mass;
                let inv_inertia = chassis.mprops.effective_world_inv_inertia;

                loop {
                    let n = candidates.len();
                    let mut a = DMatrix::<Real>::zeros(n, n);
                    let mut b = DVector::<Real>::zeros(n);

                    for i in 0..n {
                        let w_i = &candidates[i];
                        let r_i = w_i.contact - com;
                        let rxn_i = r_i.gcross(w_i.normal);
                        let i_rxn_i = inv_inertia * rxn_i;
                        b[i] = -w_i.rel_vel;

                        for j in 0..n {
                            let w_j = &candidates[j];
                            let r_j = w_j.contact - com;
                            let rxn_j = r_j.gcross(w_j.normal);
                            let lin = inv_mass * w_i.normal.dot(&w_j.normal);
                            let ang = i_rxn_i.gdot(rxn_j);
                            a[(i, j)] = lin + ang;
                        }
                    }

                    let lambda = match a.clone().lu().solve(&b) {
                        Some(l) => l,
                        // Singular (e.g. duplicate contacts). Fall back to
                        // per-wheel PGS iteration so we still do something
                        // sensible.
                        None => {
                            for w in &candidates {
                                let denom = a[(0, 0)].max(1.0e-6);
                                let lambda_i = (-w.rel_vel) / denom;
                                if lambda_i > 0.0 {
                                    chassis.apply_impulse_at_point(
                                        w.normal * lambda_i,
                                        w.contact,
                                        false,
                                    );
                                }
                            }
                            break;
                        }
                    };

                    // Active-set: if any λ_i < 0, the constraint set is
                    // infeasible as-is. Drop the most-negative wheel and
                    // resolve. The dropped wheel's inward velocity is
                    // satisfied "for free" by the other wheels' impulses.
                    let mut worst = None;
                    for i in 0..n {
                        if lambda[i] < 0.0 {
                            match worst {
                                Some((_, worst_val)) if lambda[i] >= worst_val => {}
                                _ => worst = Some((i, lambda[i])),
                            }
                        }
                    }
                    if let Some((i, _)) = worst {
                        candidates.remove(i);
                        if candidates.is_empty() {
                            break;
                        }
                        continue;
                    }

                    for i in 0..n {
                        let w = &candidates[i];
                        chassis.apply_impulse_at_point(w.normal * lambda[i], w.contact, false);
                    }
                    break;
                }
            }
        }

        // Bump-stop position correction — aggregated across all bottomed
        // wheels (PhysX Vehicle SDK pattern): one correction on the chassis
        // actor per step, resolving the worst overshoot in a single step
        // (β = 1.0 / hard-constraint behaviour). Aggregating to one
        // correction is what makes β = 1.0 safe; per-wheel correction at
        // that factor stacks additively and launches the chassis.
        //
        // Chassis rotation is driven by the per-wheel velocity impulses
        // above — each `apply_impulse_at_point` naturally distributes
        // into linear + angular response, so uneven bottom-outs on one
        // side of the vehicle pitch/roll the chassis correctly over the
        // next few steps. This single-correction position teleport only
        // handles the linear component.
        let mut deepest_overshoot = 0.0;
        let mut correction_normal = Vector::zeros();
        for wheel in &self.wheels {
            if wheel.shape_cast_info.bottomed_out
                && wheel.shape_cast_info.bottom_out_overshoot > deepest_overshoot
            {
                deepest_overshoot = wheel.shape_cast_info.bottom_out_overshoot;
                correction_normal = wheel.shape_cast_info.contact_normal_ws;
            }
        }
        if deepest_overshoot > 0.0 {
            // Cap the per-tick teleport to the wheel's max suspension travel.
            // A real chassis-into-terrain bottom-out gets corrected over a
            // few frames instead of one — visually identical, gentler on
            // velocity. A spurious bottom-out (cylinder cast clipping a
            // phantom seam between adjacent static colliders, common with
            // greedy-meshed voxel walls) only nudges the chassis a few cm
            // before the next frame re-evaluates and the spurious hit
            // usually disappears — no cascading energy injection.
            //
            // Uses the deepest-bottomed wheel's max_suspension_travel as
            // the cap. Picked because it's the natural "small but
            // meaningful" length already configured per-wheel; an
            // unbounded teleport at 60 Hz can be hundreds of m/s.
            let cap = self
                .wheels
                .iter()
                .filter(|w| w.shape_cast_info.bottomed_out)
                .map(|w| w.max_suspension_travel)
                .fold(0.0_f32, Real::max);
            let applied = deepest_overshoot.min(cap);
            let correction = correction_normal * applied;
            let new_pos = Isometry::from_parts(
                Translation::from(chassis.position().translation.vector + correction),
                chassis.position().rotation,
            );
            chassis.set_position(new_pos, true);
        }

        self.update_friction(queries.bodies, queries.colliders, dt);

        let chassis = queries
            .bodies
            .get_mut_internal_with_modification_tracking(self.chassis)
            .unwrap();

        for wheel in &mut self.wheels {
            let vel = chassis.velocity_at_point(&wheel.shape_cast_info.hard_point_ws);

            if wheel.shape_cast_info.is_in_contact {
                let mut fwd = chassis.position() * Vector::ith(self.index_forward_axis, 1.0);
                let proj = fwd.dot(&wheel.shape_cast_info.contact_normal_ws);
                fwd -= wheel.shape_cast_info.contact_normal_ws * proj;

                let proj2 = fwd.dot(&vel);

                wheel.delta_rotation = (proj2 * dt) / (wheel.radius);
                wheel.rotation += wheel.delta_rotation;
            } else {
                wheel.rotation += wheel.delta_rotation;
            }

            wheel.delta_rotation *= 0.99; //damping of rotation when not in contact
        }
    }

    /// Reference to all the wheels attached to this vehicle.
    pub fn wheels(&self) -> &[Wheel] {
        &self.wheels
    }

    /// Mutable reference to all the wheels attached to this vehicle.
    pub fn wheels_mut(&mut self) -> &mut [Wheel] {
        &mut self.wheels
    }

    fn update_suspension(&mut self, chassis_mass: Real, dt: Real) {
        // Every grounded wheel damps the SAME chassis, so the body feels the
        // sum of their dampers, not one. Applied explicitly that overshoots by
        // roughly the wheel count and the excess flips sign each step: the
        // vehicle rocks between its left and right wheels forever, unloaded
        // wheels contributing no damping to bleed it off. Folding the damper
        // into an implicit update makes it unconditionally stable for any
        // damping and wheel count, and leaves the steady state untouched — at
        // rest the scale is ~1, so ride height and feel are unchanged.
        let contacts = self
            .wheels
            .iter()
            .filter(|w| w.shape_cast_info.is_in_contact)
            .count()
            .max(1) as Real;

        for w_it in 0..self.wheels.len() {
            let wheels = &mut self.wheels[w_it];

            if wheels.shape_cast_info.is_in_contact {
                let mut force;
                //	Spring
                {
                    let rest_length = wheels.suspension_rest_length;
                    let current_length = wheels.shape_cast_info.suspension_length;
                    let length_diff = rest_length - current_length;

                    force = wheels.suspension_stiffness
                        * length_diff
                        * wheels.clipped_inv_contact_dot_suspension;
                }

                // Damper
                {
                    let projected_rel_vel = wheels.suspension_relative_velocity;
                    {
                        let susp_damping = if projected_rel_vel < 0.0 {
                            wheels.damping_compression
                        } else {
                            wheels.damping_relaxation
                        };
                        let implicit = 1.0 / (1.0 + dt * susp_damping * contacts);
                        force -= susp_damping * projected_rel_vel * implicit;
                    }
                }

                // RESULT
                wheels.wheel_suspension_force = (force * chassis_mass).max(0.0);
            } else {
                wheels.wheel_suspension_force = 0.0;
            }
        }
    }

    #[profiling::function]
    fn update_friction(&mut self, bodies: &mut RigidBodySet, colliders: &ColliderSet, dt: Real) {
        let num_wheels = self.wheels.len();

        if num_wheels == 0 {
            return;
        }

        // Below this chassis speed (m/s) a wheel with no drive/brake engages
        // longitudinal static friction (see the rolling-friction branch).
        // Above it the wheel coasts freely so neutral roll/momentum is kept.
        const STATIC_FRICTION_MAX_SPEED: Real = 0.5;
        // Fraction of the anchor drift corrected per step. Full correction (1.0)
        // fights the rest of the solver and rings; this is the usual Baumgarte
        // trade — firm enough that drift cannot accumulate, soft enough to stay
        // quiet at rest.
        const STATIC_FRICTION_ERP: Real = 0.2;
        // Furthest a stuck contact may drift before the tire is treated as
        // having slipped and re-grips at its current position.
        const MAX_STATIC_FRICTION_SLIP: Real = 0.05;
        let vehicle_speed = self.current_vehicle_speed.abs();

        self.forward_ws.resize(num_wheels, Default::default());
        self.axle.resize(num_wheels, Default::default());

        let mut num_wheels_on_ground = 0;

        //TODO: collapse all those loops into one!
        for wheel in &mut self.wheels {
            let ground_object = wheel.shape_cast_info.ground_object;

            if ground_object.is_some() {
                num_wheels_on_ground += 1;
            }

            // Latch (or release) the point this wheel is stuck to. Decided once
            // here because BOTH friction axes need it: longitudinal and lateral
            // are the same velocity-only constraint and drift the same way.
            //
            // A driven wheel is meant to travel, and a world-space anchor on a
            // moving body is meaningless, so neither gets one.
            let can_stick = ground_object.is_some()
                && wheel.engine_force == 0.0
                && vehicle_speed < STATIC_FRICTION_MAX_SPEED
                && ground_object
                    .and_then(|h| colliders[h].parent())
                    .map(|h| !bodies[h].is_dynamic())
                    .unwrap_or(true);

            if can_stick {
                let contact = wheel.shape_cast_info.contact_point_ws;
                let anchor = *wheel.static_friction_anchor.get_or_insert(contact);

                // A tire sticks only up to a finite slip distance; past that it
                // has broken traction and grips somewhere new. Without this the
                // anchor is a world point the wheel can drift arbitrarily far
                // from, and the correction stops resisting motion and starts
                // hauling the vehicle back to where it stood seconds ago —
                // slow-moving cars get visibly dragged and shoved.
                if (contact - anchor).norm() > MAX_STATIC_FRICTION_SLIP {
                    wheel.static_friction_anchor = Some(contact);
                }
            } else {
                wheel.static_friction_anchor = None;
            }

            wheel.side_impulse = 0.0;
            wheel.forward_impulse = 0.0;
        }

        {
            for i in 0..num_wheels {
                let wheel = &mut self.wheels[i];
                let ground_object = wheel.shape_cast_info.ground_object;

                if ground_object.is_some() {
                    self.axle[i] = wheel.wheel_axle_ws;

                    // Use the ground normal (actual surface), not the
                    // suspension-axial normal — otherwise forward_ws tilts
                    // with the chassis and drive force lifts the vehicle.
                    let surf_normal_ws = wheel.shape_cast_info.ground_normal_ws;
                    let proj = self.axle[i].dot(&surf_normal_ws);
                    self.axle[i] -= surf_normal_ws * proj;
                    self.axle[i] = self.axle[i]
                        .try_normalize(1.0e-5)
                        .unwrap_or_else(Vector::zeros);
                    self.forward_ws[i] = surf_normal_ws
                        .cross(&self.axle[i])
                        .try_normalize(1.0e-5)
                        .unwrap_or_else(Vector::zeros);

                    // How far the anchored contact has slid sideways, expressed
                    // as the velocity needed to undo it this step.
                    let lateral_bias = match wheel.static_friction_anchor {
                        Some(anchor) => {
                            let slid = wheel.shape_cast_info.contact_point_ws - anchor;
                            slid.dot(&self.axle[i]) * STATIC_FRICTION_ERP / dt
                        }
                        None => 0.0,
                    };

                    if let Some(ground_body) = ground_object
                        .and_then(|h| colliders[h].parent())
                        .map(|h| &bodies[h])
                        .filter(|b| b.is_dynamic())
                    {
                        wheel.side_impulse = resolve_single_bilateral(
                            &bodies[self.chassis],
                            &wheel.shape_cast_info.contact_point_ws,
                            ground_body,
                            &wheel.shape_cast_info.contact_point_ws,
                            &self.axle[i],
                            0.0,
                        );
                    } else {
                        wheel.side_impulse = resolve_single_unilateral(
                            &bodies[self.chassis],
                            &wheel.shape_cast_info.contact_point_ws,
                            &self.axle[i],
                            lateral_bias,
                        );
                    }

wheel.side_impulse *= wheel.side_friction_stiffness;

                    // Side friction is solved Jacobi-style: every grounded wheel
                    // computes its impulse against the same pre-impulse velocity,
                    // then all are applied together. N wheels then over-correct
                    // the shared lateral/roll DOF by ~N×. The 0.2 resolver
                    // relaxation alone keeps that stable only up to a net
                    // per-wheel gain of 1.0 — which side_friction_stiffness=5.0
                    // (0.2 × 5.0) sits exactly at, so multiple wheels tip it into
                    // a standing oscillation (the at-rest lateral buzz). Share the
                    // cancel across wheels, exactly as calc_rolling_friction does
                    // for longitudinal friction, so the summed impulse stays
                    // ≈ critical (gain 1) for any wheel count without changing
                    // circle-limited peak grip.
                    wheel.side_impulse /= num_wheels_on_ground as Real;
                }
            }
        }

        let side_factor = 1.0;
        let fwd_factor = 0.5;

        let mut sliding = false;
        {
            for wheel_id in 0..num_wheels {
                let wheel = &mut self.wheels[wheel_id];
                let ground_object = wheel.shape_cast_info.ground_object;

                let mut rolling_friction = 0.0;

                if ground_object.is_some() {
                    if wheel.engine_force != 0.0 {
                        rolling_friction = wheel.engine_force * dt;
                    } else {
                        // Static friction at rest. With no drive or brake,
                        // Bullet's raycast model zeroes rolling friction so the
                        // vehicle coasts — but that leaves a parked wheel a free
                        // longitudinal roller, so the horizontal component of an
                        // axial suspension force on a pitched chassis creeps the
                        // vehicle with nothing to resist it. When the vehicle is
                        // nearly stopped, hold the wheel up to the tire's
                        // friction limit (μ·N) so it sticks longitudinally just
                        // like the lateral axis already does; above the speed
                        // gate the impulse stays 0 and the wheel coasts.
                        // Braking and static friction are not alternatives. A
                        // stopped wheel resists with whichever is stronger: the
                        // brake's grip, or the tire's own μN. Treating them as
                        // exclusive is what let creep survive — callers park a
                        // token brake on idle wheels, which always won this
                        // branch and capped the hold at that token value while
                        // the far larger static limit went unused.
                        let sticking = vehicle_speed < STATIC_FRICTION_MAX_SPEED;

                        let static_limit = if sticking {
                            wheel.wheel_suspension_force * dt * wheel.friction_slip
                        } else {
                            0.0
                        };
                        let max_impulse = wheel.brake.max(static_limit);

                        // Latch the contact where it first stuck, then measure
                        // how far it has slid along the travel direction since.
                        // Only static ground gets an anchor: a world-space point
                        // on a moving body is meaningless, and the creep this
                        // exists to kill is against terrain.
                        let bias_velocity = match wheel.static_friction_anchor {
                            Some(anchor) => {
                                let slid = wheel.shape_cast_info.contact_point_ws - anchor;
                                slid.dot(&self.forward_ws[wheel_id]) * STATIC_FRICTION_ERP / dt
                            }
                            None => 0.0,
                        };

                        let contact_pt = WheelContactPoint::new(
                            &bodies[self.chassis],
                            ground_object
                                .and_then(|h| colliders[h].parent())
                                .map(|h| &bodies[h]),
                            wheel.shape_cast_info.contact_point_ws,
                            self.forward_ws[wheel_id],
                            max_impulse,
                        );
                        assert!(num_wheels_on_ground > 0);
                        rolling_friction =
                            contact_pt.calc_rolling_friction(num_wheels_on_ground, bias_velocity);

                        // Saturating the limit means the tire broke traction —
                        // it is sliding, so the old anchor no longer describes
                        // where it is stuck. Re-latch here rather than dragging
                        // a stale point behind a sliding wheel.
                        if wheel.static_friction_anchor.is_some()
                            && rolling_friction.abs() >= max_impulse
                        {
                            wheel.static_friction_anchor =
                                Some(wheel.shape_cast_info.contact_point_ws);
                        }
                    }
                }

                //switch between active rolling (throttle), braking and non-active rolling friction (no throttle/break)

                wheel.forward_impulse = 0.0;
                wheel.skid_info = 1.0;

                if ground_object.is_some() {
                    let max_imp = wheel.wheel_suspension_force * dt * wheel.friction_slip;
                    let max_imp_side = max_imp;
                    let max_imp_squared = max_imp * max_imp_side;
                    assert!(max_imp_squared >= 0.0);

                    wheel.forward_impulse = rolling_friction;

                    let x = wheel.forward_impulse * fwd_factor;
                    let y = wheel.side_impulse * side_factor;

                    let impulse_squared = x * x + y * y;

                    if impulse_squared > max_imp_squared {
                        sliding = true;

                        let factor = max_imp * crate::utils::inv(impulse_squared.sqrt());
                        wheel.skid_info *= factor;
                    }
                }
            }
        }

        if sliding {
            for wheel in &mut self.wheels {
                if wheel.side_impulse != 0.0 && wheel.skid_info < 1.0 {
                    wheel.forward_impulse *= wheel.skid_info;
                    wheel.side_impulse *= wheel.skid_info;
                }
            }
        }

        // apply the impulses
        {
            let chassis = bodies
                .get_mut_internal_with_modification_tracking(self.chassis)
                .unwrap();

            for wheel_id in 0..num_wheels {
                let wheel = &self.wheels[wheel_id];
                let contact_point = wheel.shape_cast_info.contact_point_ws;

                // Anti-squat / anti-roll shifts: both slide the impulse
                // application point UP along the chassis-up axis toward COM
                // height, killing the pitch/roll moment generated by forces
                // applied at the bottom of the wheel.
                let chassis_up =
                    chassis.position().rotation * Vector::ith(self.index_up_axis, 1.0);
                let to_contact_up = chassis_up.dot(&(contact_point - chassis.center_of_mass()));

                if wheel.forward_impulse != 0.0 {
                    // `anti_squat ∈ [0,1]`: 0 = at contact (Bullet), 1 = at COM height.
                    let forward_point = contact_point - chassis_up * (to_contact_up * wheel.anti_squat);
                    chassis.apply_impulse_at_point(
                        self.forward_ws[wheel_id] * wheel.forward_impulse,
                        forward_point,
                        false,
                    );
                }
                if wheel.side_impulse != 0.0 {
                    let side_impulse = self.axle[wheel_id] * wheel.side_impulse;

                    let side_point =
                        contact_point - chassis_up * (to_contact_up * (1.0 - wheel.roll_influence));

                    chassis.apply_impulse_at_point(side_impulse, side_point, false);

                    // TODO: apply friction impulse on the ground
                    // let ground_object = self.wheels[wheel_id].shape_cast_info.ground_object;
                    // ground_object.apply_impulse_at_point(
                    //     -side_impulse,
                    //     wheels.shape_cast_info.contact_point_ws,
                    //     false,
                    // );
                }
            }
        }
    }
}

struct WheelContactPoint<'a> {
    body0: &'a RigidBody,
    body1: Option<&'a RigidBody>,
    friction_position_world: Point<Real>,
    friction_direction_world: Vector<Real>,
    jac_diag_ab_inv: Real,
    max_impulse: Real,
}

impl<'a> WheelContactPoint<'a> {
    pub fn new(
        body0: &'a RigidBody,
        body1: Option<&'a RigidBody>,
        friction_position_world: Point<Real>,
        friction_direction_world: Vector<Real>,
        max_impulse: Real,
    ) -> Self {
        fn impulse_denominator(body: &RigidBody, pos: &Point<Real>, n: &Vector<Real>) -> Real {
            let dpt = pos - body.center_of_mass();
            let gcross = dpt.gcross(*n);
            let v = (body.mprops.effective_world_inv_inertia * gcross).gcross(dpt);
            // TODO: take the effective inv mass into account instead of the inv_mass?
            body.mprops.local_mprops.inv_mass + n.dot(&v)
        }
        let denom0 =
            impulse_denominator(body0, &friction_position_world, &friction_direction_world);
        let denom1 = body1
            .map(|body1| {
                impulse_denominator(body1, &friction_position_world, &friction_direction_world)
            })
            .unwrap_or(0.0);
        let relaxation = 1.0;
        let jac_diag_ab_inv = relaxation / (denom0 + denom1);

        Self {
            body0,
            body1,
            friction_position_world,
            friction_direction_world,
            jac_diag_ab_inv,
            max_impulse,
        }
    }

    /// Impulse that cancels the relative velocity along the friction
    /// direction, plus `bias_velocity` — the rate at which an anchored
    /// contact must travel to undo the drift it has already accumulated.
    /// Without that term this is a pure velocity constraint and position
    /// error accrues forever (see `Wheel::static_friction_anchor`).
    pub fn calc_rolling_friction(
        &self,
        num_wheels_on_ground: usize,
        bias_velocity: Real,
    ) -> Real {
        let contact_pos_world = self.friction_position_world;
        let max_impulse = self.max_impulse;

        let vel1 = self.body0.velocity_at_point(&contact_pos_world);
        let vel2 = self
            .body1
            .map(|b| b.velocity_at_point(&contact_pos_world))
            .unwrap_or_else(Vector::zeros);
        let vel = vel1 - vel2;
        let vrel = self.friction_direction_world.dot(&vel);

        // friction that moves us to zero relative velocity AND unwinds the
        // drift the anchor has recorded
        (-(vrel + bias_velocity) * self.jac_diag_ab_inv / (num_wheels_on_ground as Real))
            .clamp(-max_impulse, max_impulse)
    }
}

fn resolve_single_bilateral(
    body1: &RigidBody,
    pt1: &Point<Real>,
    body2: &RigidBody,
    pt2: &Point<Real>,
    normal: &Vector<Real>,
    bias_velocity: Real,
) -> Real {
    let vel1 = body1.velocity_at_point(pt1);
    let vel2 = body2.velocity_at_point(pt2);
    let dvel = vel1 - vel2;

    let dpt1 = pt1 - body1.center_of_mass();
    let dpt2 = pt2 - body2.center_of_mass();
    let aj = dpt1.gcross(*normal);
    let bj = dpt2.gcross(-*normal);
    let iaj = body1.mprops.effective_world_inv_inertia * aj;
    let ibj = body2.mprops.effective_world_inv_inertia * bj;

    // TODO: take the effective_inv_mass into account?
    let im1 = body1.mprops.local_mprops.inv_mass;
    let im2 = body2.mprops.local_mprops.inv_mass;

    let jac_diag_ab = im1 + im2 + iaj.gdot(iaj) + ibj.gdot(ibj);
    let jac_diag_ab_inv = crate::utils::inv(jac_diag_ab);
    let rel_vel = normal.dot(&dvel);

    //todo: move this into proper structure
    let contact_damping = 0.2;
    -contact_damping * (rel_vel + bias_velocity) * jac_diag_ab_inv
}

fn resolve_single_unilateral(
    body1: &RigidBody,
    pt1: &Point<Real>,
    normal: &Vector<Real>,
    bias_velocity: Real,
) -> Real {
    let vel1 = body1.velocity_at_point(pt1);
    let dvel = vel1;
    let dpt1 = pt1 - body1.center_of_mass();
    let aj = dpt1.gcross(*normal);
    let iaj = body1.mprops.effective_world_inv_inertia * aj;

    // TODO: take the effective_inv_mass into account?
    let im1 = body1.mprops.local_mprops.inv_mass;
    let jac_diag_ab = im1 + iaj.gdot(iaj);
    let jac_diag_ab_inv = crate::utils::inv(jac_diag_ab);
    let rel_vel = normal.dot(&dvel);

    //todo: move this into proper structure
    let contact_damping = 0.2;
    -contact_damping * (rel_vel + bias_velocity) * jac_diag_ab_inv
}
