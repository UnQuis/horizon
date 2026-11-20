//! Powder snow collision over entity metadata and lazily queried walkability.
use crate::{collision_context::CollisionContext, entity_obstruction::EntityIdentity};
use horizon_core::{
    geometry::{Aabb, Vec3, VoxelShape},
    position::BlockPos,
};
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug)]
pub struct PowderSnowEntity {
    pub fall_distance: f64,
    pub falling_block: bool,
}

pub trait PowderSnowWorld {
    type Error;
    /// Snapshot of the entity's fallDistance field and FallingBlockEntity type.
    fn powder_snow_entity(
        &mut self,
        entity: EntityIdentity,
    ) -> Result<PowderSnowEntity, Self::Error>;
    fn powder_snow_walkable_tag(&mut self, entity: EntityIdentity) -> Result<bool, Self::Error>;
    /// Invoke only for a living entity; item identity follows ItemStack.is,
    /// including its empty-stack normalization.
    fn feet_are_leather_boots(&mut self, entity: EntityIdentity) -> Result<bool, Self::Error>;
    /// A mod's answer on whether the entity can walk on the powder snow at
    /// `pos`, with vanilla's `can_walk` as the default. The implementation that
    /// can reach native mods asks them here; without a listener it returns the
    /// value it was given. The default keeps a mod-free world bit-for-bit the same.
    fn powder_snow_mod_walkable(
        &mut self,
        pos: BlockPos,
        can_walk: bool,
    ) -> Result<bool, Self::Error> {
        let _ = pos;
        Ok(can_walk)
    }
}

pub fn can_walk_on_powder_snow<W: PowderSnowWorld>(
    world: &mut W,
    entity: EntityIdentity,
    living: bool,
) -> Result<bool, W::Error> {
    if world.powder_snow_walkable_tag(entity)? {
        return Ok(true);
    }
    if living {
        return world.feet_are_leather_boots(entity);
    }
    Ok(false)
}

pub fn powder_snow_collision_shape<W: PowderSnowWorld>(
    world: &mut W,
    position: BlockPos,
    context: &CollisionContext<'_>,
) -> Result<VoxelShape, W::Error> {
    if context.is_placement() {
        return Ok(VoxelShape::empty());
    }
    let Some(entity) = context.entity() else {
        return Ok(VoxelShape::empty());
    };
    let snapshot = world.powder_snow_entity(entity)?;
    if snapshot.fall_distance > 2.5 {
        static FALLING: OnceLock<VoxelShape> = OnceLock::new();
        return Ok(FALLING
            .get_or_init(|| {
                VoxelShape::from_box(Aabb::new(
                    Vec3::ZERO,
                    Vec3::new(1.0, f64::from(0.9f32), 1.0),
                ))
                .expect("bounded falling powder snow shape")
            })
            .clone());
    }
    if snapshot.falling_block
        || ({
            let vanilla =
                can_walk_on_powder_snow(world, entity, context.living_entity().is_some())?;
            world.powder_snow_mod_walkable(position, vanilla)?
        } && context.is_above(&VoxelShape::block(), position, false)
            && !context.is_descending())
    {
        return Ok(VoxelShape::block());
    }
    Ok(VoxelShape::empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct World {
        override_answer: Option<bool>,
        asked: Vec<BlockPos>,
    }
    impl PowderSnowWorld for World {
        type Error = std::convert::Infallible;
        fn powder_snow_entity(
            &mut self,
            _: EntityIdentity,
        ) -> Result<PowderSnowEntity, Self::Error> {
            Ok(PowderSnowEntity {
                fall_distance: 0.0,
                falling_block: false,
            })
        }
        fn powder_snow_walkable_tag(&mut self, _: EntityIdentity) -> Result<bool, Self::Error> {
            Ok(false)
        }
        fn feet_are_leather_boots(&mut self, _: EntityIdentity) -> Result<bool, Self::Error> {
            Ok(false)
        }
        fn powder_snow_mod_walkable(
            &mut self,
            pos: BlockPos,
            can_walk: bool,
        ) -> Result<bool, Self::Error> {
            self.asked.push(pos);
            Ok(self.override_answer.unwrap_or(can_walk))
        }
    }

    /// The mod point is asked with the block's own position, and its pass-through
    /// default leaves vanilla's answer alone.
    #[test]
    fn mod_walkable_passes_through_and_overrides_at_the_snow_position() {
        let at = BlockPos::new(3, 64, 5);
        let mut world = World {
            override_answer: None,
            asked: Vec::new(),
        };
        assert!(world.powder_snow_mod_walkable(at, true).unwrap());
        assert!(!world.powder_snow_mod_walkable(at, false).unwrap());
        assert_eq!(world.asked, vec![at, at]);

        let mut overridden = World {
            override_answer: Some(true),
            asked: Vec::new(),
        };
        assert!(overridden.powder_snow_mod_walkable(at, false).unwrap());
    }

    /// An implementation that does not override the method keeps vanilla's answer.
    #[test]
    fn a_world_without_a_mod_point_keeps_vanilla() {
        struct Plain;
        impl PowderSnowWorld for Plain {
            type Error = std::convert::Infallible;
            fn powder_snow_entity(
                &mut self,
                _: EntityIdentity,
            ) -> Result<PowderSnowEntity, Self::Error> {
                unreachable!()
            }
            fn powder_snow_walkable_tag(&mut self, _: EntityIdentity) -> Result<bool, Self::Error> {
                unreachable!()
            }
            fn feet_are_leather_boots(&mut self, _: EntityIdentity) -> Result<bool, Self::Error> {
                unreachable!()
            }
        }
        let mut plain = Plain;
        assert!(
            plain
                .powder_snow_mod_walkable(BlockPos::new(0, 0, 0), true)
                .unwrap()
        );
        assert!(
            !plain
                .powder_snow_mod_walkable(BlockPos::new(0, 0, 0), false)
                .unwrap()
        );
    }
}
