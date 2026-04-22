//! Utilities for controlling the trajectories of objects in a non-physical way.

pub use self::character_controller::{
    CharacterAutostep, CharacterCollision, CharacterLength, EffectiveCharacterMovement,
    KinematicCharacterController,
};
pub use self::pid_controller::{PdController, PdErrors, PidController};

#[cfg(feature = "dim3")]
pub use self::ray_cast_vehicle_controller::{DynamicRayCastVehicleController, Wheel, WheelTuning};
#[cfg(feature = "dim3")]
pub use self::shape_cast_vehicle_controller::{
    DynamicShapeCastVehicleController, ShapeCastInfo,
    Wheel as ShapeCastWheel, WheelTuning as ShapeCastWheelTuning,
};

mod character_controller;

mod pid_controller;
#[cfg(feature = "dim3")]
mod ray_cast_vehicle_controller;
#[cfg(feature = "dim3")]
mod shape_cast_vehicle_controller;
