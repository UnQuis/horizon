use super::*;

/// The hook points. Each only forwards to the mod API.
impl Game<'_> {
    #[inline(never)]
    pub(in crate::game) fn native_mod_slot_will_change(
        &self,
        holder: EntityRef,
        slot: &str,
        index: usize,
        cosmetic: bool,
        old: Option<&ItemStack>,
        new: Option<&ItemStack>,
        source: &str,
        cancel: &mut bool,
    ) {
        self.fire_mod_slot_will_change(holder, slot, index, cosmetic, old, new, source, cancel);
    }

    #[inline(never)]
    pub(in crate::game) fn native_mod_slot_changed(
        &self,
        holder: EntityRef,
        slot: &str,
        index: usize,
        cosmetic: bool,
        old: Option<&ItemStack>,
        new: Option<&ItemStack>,
        source: &str,
    ) {
        self.fire_mod_slot_changed(holder, slot, index, cosmetic, old, new, source);
    }
}

/// The host helpers a mod calls at its own two moments.
impl Game<'_> {
    pub(in crate::game) fn mod_slot_will_change(
        &mut self,
        holder: EntityRef,
        slot: &str,
        index: usize,
        cosmetic: bool,
        old: Option<ItemStack>,
        new: Option<ItemStack>,
        source: &str,
    ) -> bool {
        let mut cancel = false;
        mod_api::item::mod_slots::native_mod_slot_will_change(
            self,
            holder,
            slot,
            index,
            cosmetic,
            old.as_ref(),
            new.as_ref(),
            source,
            &mut cancel,
        );
        !cancel
    }

    pub(in crate::game) fn mod_slot_changed(
        &mut self,
        holder: EntityRef,
        slot: &str,
        index: usize,
        cosmetic: bool,
        old: Option<ItemStack>,
        new: Option<ItemStack>,
        source: &str,
    ) {
        mod_api::item::mod_slots::native_mod_slot_changed(
            self,
            holder,
            slot,
            index,
            cosmetic,
            old.as_ref(),
            new.as_ref(),
            source,
        );
    }
}
