use super::*;

/// The hook points. Each does nothing in Horizon.
impl Game<'_> {
    /// The game's target for `get_enchantment_level_for`: a thin forwarder
    /// that passes an immutable host and the caller's owner and place to the
    /// mod API's wearer-aware entry point. The engine never calls this on its
    /// own; a mod calls that entry point from its own equip moment.
    #[inline(never)]
    pub(in crate::game) fn native_get_enchantment_level_for(
        &self,
        stack: &ItemStack,
        target: Option<&Identifier>,
        owner: Option<EntityRef>,
        mod_slot: Option<&str>,
        enchantments: &mut ItemEnchantments,
    ) {
        mod_api::item::enchantments::native_get_enchantment_level_for(
            self,
            stack,
            target,
            owner,
            mod_slot,
            enchantments,
        );
    }
}
