//! The target of `get_enchantment_level` (`GetEnchantmentLevelEvent`).
use crate::item::item_stack::ItemStack;
#[allow(unused_imports)]
use crate::*;
/// `GetEnchantmentLevelEvent`: the enchantments gameplay sees on `stack`,
/// starting from its own `minecraft:enchantments`. `target` is the one
/// enchantment a read asks for (`ItemStack.getEnchantmentLevel`), and then
/// only its level counts; `None` when every one is read
/// (`ItemStack.getAllEnchantments`). Set, upgrade or remove levels in
/// `enchantments`: what is left is what fights, mining, fishing, tridents,
/// crossbows, armor, loot and every other enchantment effect use. Tooltips,
/// the anvil, the grindstone, the enchanting table, saves and the network
/// keep the stack's own component. Asked on every such read, often many
/// times a tick, on the server and on a client (its tooltip modifiers).
///
/// The compatibility entry point: a stack read with no known wearer. It asks
/// the hook with `owner` and `mod_slot` unset, exactly as it did before those
/// fields existed. A mod that knows the wearer calls
/// [`native_get_enchantment_level_for`] instead.
#[inline(never)]
pub fn native_get_enchantment_level(
    stack: &ItemStack,
    target: Option<&Identifier>,
    enchantments: &mut ItemEnchantments,
) {
    native_get_enchantment_level_for(stack, target, None, None, enchantments);
}

/// The wearer-aware entry point: the same hook with `owner`, the entity
/// wearing or holding `stack`, and `mod_slot`, the caller's own place name
/// (`curios:ring`), set by the caller. Both are `None` for an ordinary stack
/// read such as a tooltip or a save; this function itself does not discover
/// a wearer. A mod must call it from the moment it knows one to give a worn
/// item a per-wearer level (Fortune or Looting on a curio). `None` for either
/// is accepted and leaves the event's field empty.
#[inline(never)]
pub fn native_get_enchantment_level_for(
    stack: &ItemStack,
    target: Option<&Identifier>,
    owner: Option<EntityRef>,
    mod_slot: Option<&str>,
    enchantments: &mut ItemEnchantments,
) {
    crate::dispatch_get_enchantment_level(stack, target, owner, mod_slot, enchantments);
}
