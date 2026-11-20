use super::*;

/// The hook points. Each does nothing in Horizon.
impl Game<'_> {
    /// The game's target for `get_enchantment_level`, so the generated event
    /// can carry `owner` and `mod_slot` where a caller knows them. Ordinary
    /// stack reads reach the hook through the unchanged public three-argument
    /// `item::enchantments::native_get_enchantment_level`, which passes both as
    /// `None`. A mod that knows the wearer calls
    /// `item::enchantments::native_get_enchantment_level_for` instead.
    #[inline(never)]
    pub(in crate::game) fn native_get_enchantment_level(
        &self,
        stack: &ItemStack,
        target: Option<&Identifier>,
        owner: Option<EntityRef>,
        mod_slot: Option<&str>,
        enchantments: &mut ItemEnchantments,
    ) {
        self.fire_get_enchantment_level(stack, target, owner, mod_slot, enchantments);
    }
}
