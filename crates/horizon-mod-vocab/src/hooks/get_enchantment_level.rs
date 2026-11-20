//! The `get_enchantment_level` hooks and helpers: declarations the mod API generator reads (see `hooks/mod.rs`).

pub trait Hooks {
    /// `GetEnchantmentLevelEvent`: the enchantments gameplay sees on `stack`,
    /// starting from its own `minecraft:enchantments`. `target` is the one
    /// enchantment a read asks for (`ItemStack.getEnchantmentLevel`), and then
    /// only its level counts; `None` when every one is read
    /// (`ItemStack.getAllEnchantments`). Set, upgrade or remove levels in
    /// `enchantments`: what is left is what fights, mining, fishing, tridents,
    /// crossbows, armor, loot and every other enchantment effect use. Tooltips,
    /// the anvil, the grindstone, the enchanting table, saves and the network
    /// keep the stack's own component. `owner` is the entity wearing or
    /// holding the stack where the caller knows one, `None` on ordinary stack
    /// reads; `mod_slot` is the caller's own place name for it (`curios:ring`),
    /// `None` on a vanilla slot. A mod can use both to give a worn item a
    /// per-wearer level (Fortune or Looting on a curio), and a mod that knows
    /// the wearer can call `item::enchantments::native_get_enchantment_level`
    /// with them directly. Asked on every such read, often many times a tick,
    /// on the server and on a client (its tooltip modifiers).
    #[hook(
        side = "both",
        contract = "listen",
        neoforge = "GetEnchantmentLevelEvent",
        target = "mod_api::item::enchantments::native_get_enchantment_level"
    )]
    fn get_enchantment_level();
}
