//! Engine-stack callbacks installed by the game at its API boundary.
use horizon_items::items::ItemStack;
use horizon_world::native_load_context::LoopContext;
use mod_api::{EntityRef, Identifier, ItemAttributeModifiers, ItemEnchantments};
use std::sync::OnceLock;

/// Stack-bearing hooks whose API snapshots are made by the game.
pub struct StackHooks {
    pub ingredient_test:
        fn(&LoopContext, &str, &horizon_core::nbt::Compound, &ItemStack, &mut bool),
    pub enchantments:
        fn(&ItemStack, Option<&Identifier>, Option<EntityRef>, Option<&str>, &mut ItemEnchantments),
    pub attributes: fn(&ItemStack, &mut ItemAttributeModifiers),
}
static HOOKS: OnceLock<StackHooks> = OnceLock::new();
/// Install the game conversion callbacks once.
pub fn install(hooks: StackHooks) -> bool {
    HOOKS.set(hooks).is_ok()
}
pub(crate) fn ingredient_test(
    context: &LoopContext,
    kind: &str,
    data: &horizon_core::nbt::Compound,
    stack: &ItemStack,
    accept: &mut bool,
) {
    if let Some(hooks) = HOOKS.get() {
        (hooks.ingredient_test)(context, kind, data, stack, accept);
    }
}
pub(crate) fn enchantments(
    stack: &ItemStack,
    target: Option<&Identifier>,
    owner: Option<EntityRef>,
    mod_slot: Option<&str>,
    value: &mut ItemEnchantments,
) {
    if let Some(hooks) = HOOKS.get() {
        (hooks.enchantments)(stack, target, owner, mod_slot, value);
    }
}
pub(crate) fn attributes(stack: &ItemStack, value: &mut ItemAttributeModifiers) {
    if let Some(hooks) = HOOKS.get() {
        (hooks.attributes)(stack, value);
    }
}
