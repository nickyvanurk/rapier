//! A vehicle controller based on cylinder shape-casting, ported and modified from Bullet’s `btRaycastVehicle`.

use crate::dynamics::{RigidBody, RigidBodyHandle, RigidBodySet};
use crate::geometry::{ColliderHandle, ColliderSet, Cylinder};
use crate::math::{Isometry, Point, Real, Rotation, Translation, Vector};
use crate::pipeline::QueryPipeline;
use crate::prelude::QueryPipelineMut;
use crate::utils::{SimdCross, SimdDot};
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
    /// The (world-space) contact normal between the wheel and the floor.
    pub contact_normal_ws: Vector<Real>,
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
    fn shape_cast(&mut self, queries: &QueryPipeline, chassis: &RigidBody, wheel_id: usize) {
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

        // Start the cylinder one radius behind the hard_point (opposite to
        // cast direction) so its near cap sits at the hard_point — prevents
        // starting inside the ground. If toi=0 (cylinder still clips terrain
        // at that offset), retry at 2× radius.
        // suspension_length = toi − offset, matching the raycast’s
        // hit_distance − radius formula.
        let r = wheel.radius;
        let raylen = wheel.suspension_rest_length + r;

        let (hit, offset) = {
            let offset = r;
            let pos = Isometry::from_parts(
                Translation::from(source.coords - direction * offset),
                cyl_rot,
            );
            let options = ShapeCastOptions {
                max_time_of_impact: raylen + offset,
                target_distance: 0.0,
                stop_at_penetration: true,
                compute_impact_geometry_on_penetration: true,
            };
            let result = queries.cast_shape(&pos, &direction, &cylinder, options);

            // toi=0 means the cylinder started in penetration — retry with double offset
            if result.as_ref().is_some_and(|(_, h)| h.time_of_impact == 0.0) {
                let offset = r * 2.0;
                let pos = Isometry::from_parts(
                    Translation::from(source.coords - direction * offset),
                    cyl_rot,
                );
                let options = ShapeCastOptions {
                    max_time_of_impact: raylen + offset,
                    target_distance: 0.0,
                    stop_at_penetration: true,
                    compute_impact_geometry_on_penetration: true,
                };
                (queries.cast_shape(&pos, &direction, &cylinder, options), offset)
            } else {
                (result, offset)
            }
        };

        wheel.shape_cast_info.ground_object = None;

        if let Some((collider_hit, hit)) = hit {
            // Lock the contact normal to the suspension axis. Using
            // `hit.normal1` lets a wheel clipping a vertical face (wall,
            // curb, side of a log) apply the suspension force sideways and
            // launch the chassis. Trading a bit of slope-accuracy for a
            // cleanly-axial spring.
            let normal = -wheel.wheel_direction_ws;

            wheel.shape_cast_info.contact_normal_ws = normal;
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
            wheel.shape_cast_info.bottomed_out = raw_length < 0.0;
            wheel.shape_cast_info.bottom_out_overshoot = (-raw_length).max(0.0);
            wheel.shape_cast_info.suspension_length = raw_length.clamp(0.0, max_suspension_length);
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
            // No contact, put wheel info as in rest position
            wheel.shape_cast_info.suspension_length = wheel.suspension_rest_length;
            wheel.shape_cast_info.bottomed_out = false;
            wheel.shape_cast_info.bottom_out_overshoot = 0.0;
            wheel.suspension_relative_velocity = 0.0;
            wheel.shape_cast_info.contact_normal_ws = -wheel.wheel_direction_ws;
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
            self.shape_cast(&queries.as_ref(), chassis, wheel_id);
        }

        let chassis_mass = chassis.mass();
        self.update_suspension(chassis_mass);

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

            // Bump stop. When the suspension is fully compressed the spring
            // is physically at its end of travel; further compression can't
            // be absorbed by force alone. Cancel the chassis velocity into
            // the ground at the contact (plastic collision) and shift the
            // chassis back by the overshoot so the wheel visibly sits at
            // the hard point instead of below it.
            if wheel.shape_cast_info.bottomed_out {
                let normal = wheel.shape_cast_info.contact_normal_ws;
                let contact = wheel.shape_cast_info.contact_point_ws;
                let vel = chassis.velocity_at_point(&contact);
                let rel_vel = normal.dot(&vel);
                if rel_vel < 0.0 {
                    let dpt = contact - chassis.center_of_mass();
                    let aj = dpt.gcross(normal);
                    let iaj = chassis.mprops.effective_world_inv_inertia * aj;
                    let jac = chassis.mprops.local_mprops.inv_mass + iaj.gdot(iaj);
                    let inv_jac = crate::utils::inv(jac);
                    let impulse_mag = -rel_vel * inv_jac;
                    chassis.apply_impulse_at_point(normal * impulse_mag, contact, false);
                }

                // Position correction: translate the chassis up by the
                // overshoot so we don't accumulate penetration. Use a
                // Baumgarte-style 0.8 factor to keep it stable.
                let correction = normal * (wheel.shape_cast_info.bottom_out_overshoot * 0.8);
                let new_pos = Isometry::from_parts(
                    Translation::from(chassis.position().translation.vector + correction),
                    chassis.position().rotation,
                );
                chassis.set_position(new_pos, true);
            }
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

    fn update_suspension(&mut self, chassis_mass: Real) {
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
                        force -= susp_damping * projected_rel_vel;
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

        self.forward_ws.resize(num_wheels, Default::default());
        self.axle.resize(num_wheels, Default::default());

        let mut num_wheels_on_ground = 0;

        //TODO: collapse all those loops into one!
        for wheel in &mut self.wheels {
            let ground_object = wheel.shape_cast_info.ground_object;

            if ground_object.is_some() {
                num_wheels_on_ground += 1;
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

                    let surf_normal_ws = wheel.shape_cast_info.contact_normal_ws;
                    let proj = self.axle[i].dot(&surf_normal_ws);
                    self.axle[i] -= surf_normal_ws * proj;
                    self.axle[i] = self.axle[i]
                        .try_normalize(1.0e-5)
                        .unwrap_or_else(Vector::zeros);
                    self.forward_ws[i] = surf_normal_ws
                        .cross(&self.axle[i])
                        .try_normalize(1.0e-5)
                        .unwrap_or_else(Vector::zeros);

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
                        );
                    } else {
                        wheel.side_impulse = resolve_single_unilateral(
                            &bodies[self.chassis],
                            &wheel.shape_cast_info.contact_point_ws,
                            &self.axle[i],
                        );
                    }

                    wheel.side_impulse *= wheel.side_friction_stiffness;
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
                        let default_rolling_friction_impulse = 0.0;
                        let max_impulse = if wheel.brake != 0.0 {
                            wheel.brake
                        } else {
                            default_rolling_friction_impulse
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
                        rolling_friction = contact_pt.calc_rolling_friction(num_wheels_on_ground);
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
                let mut impulse_point = wheel.shape_cast_info.contact_point_ws;

                if wheel.forward_impulse != 0.0 {
                    chassis.apply_impulse_at_point(
                        self.forward_ws[wheel_id] * wheel.forward_impulse,
                        impulse_point,
                        false,
                    );
                }
                if wheel.side_impulse != 0.0 {
                    let side_impulse = self.axle[wheel_id] * wheel.side_impulse;

                    let v_chassis_world_up =
                        chassis.position().rotation * Vector::ith(self.index_up_axis, 1.0);
                    impulse_point -= v_chassis_world_up
                        * (v_chassis_world_up.dot(&(impulse_point - chassis.center_of_mass()))
                            * (1.0 - wheel.roll_influence));

                    chassis.apply_impulse_at_point(side_impulse, impulse_point, false);

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

    pub fn calc_rolling_friction(&self, num_wheels_on_ground: usize) -> Real {
        let contact_pos_world = self.friction_position_world;
        let max_impulse = self.max_impulse;

        let vel1 = self.body0.velocity_at_point(&contact_pos_world);
        let vel2 = self
            .body1
            .map(|b| b.velocity_at_point(&contact_pos_world))
            .unwrap_or_else(Vector::zeros);
        let vel = vel1 - vel2;
        let vrel = self.friction_direction_world.dot(&vel);

        // calculate friction that moves us to zero relative velocity
        (-vrel * self.jac_diag_ab_inv / (num_wheels_on_ground as Real))
            .clamp(-max_impulse, max_impulse)
    }
}

fn resolve_single_bilateral(
    body1: &RigidBody,
    pt1: &Point<Real>,
    body2: &RigidBody,
    pt2: &Point<Real>,
    normal: &Vector<Real>,
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
    -contact_damping * rel_vel * jac_diag_ab_inv
}

fn resolve_single_unilateral(body1: &RigidBody, pt1: &Point<Real>, normal: &Vector<Real>) -> Real {
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
    -contact_damping * rel_vel * jac_diag_ab_inv
}
