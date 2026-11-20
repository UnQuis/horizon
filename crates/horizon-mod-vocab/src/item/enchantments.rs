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
/// keep the stack's own component. `owner` is the entity wearing or holding
/// the stack where the caller knows one (`None` for an ordinary stack read such
/// as a tooltip or a save), and `mod_slot` the caller's own place name for it
/// (`curios:ring`); a mod may use them to give a worn item a per-wearer level.
/// Asked on every such read, often many times a tick, on the server and on a
/// client (its tooltip modifiers).
#[inline(never)]
pub fn native_get_enchantment_level(
    stack: &ItemStack,
    target: Option<&Identifier>,
    owner: Option<EntityRef>,
    mod_slot: Option<&str>,
    enchantments: &mut ItemEnchantments,
) {
    crate::dispatch_get_enchantment_level(stack, target, owner, mod_slot, enchantments);
}
