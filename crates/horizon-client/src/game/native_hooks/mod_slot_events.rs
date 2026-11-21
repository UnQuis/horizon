use super::*;

/// The hook points. Each only forwards to the mod API.
impl Game<'_> {
    #[inline(never)]
    pub(in crate::game) fn native_mod_slot_will_change(
        &self,
        call: mod_api::__private::ModSlotWillChangeCall<'_>,
    ) {
        self.fire_mod_slot_will_change(call);
    }

    #[inline(never)]
    pub(in crate::game) fn native_mod_slot_changed(
        &self,
        call: mod_api::__private::ModSlotChangedCall<'_>,
    ) {
        self.fire_mod_slot_changed(call);
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
        old: Option<mod_api::ItemStack>,
        new: Option<mod_api::ItemStack>,
        source: &str,
    ) -> bool {
        let mut cancel = false;
        self.native_mod_slot_will_change(mod_api::__private::ModSlotWillChangeCall {
            holder,
            slot,
            index,
            cosmetic,
            old: old.as_ref(),
            new: new.as_ref(),
            source,
            cancel: &mut cancel,
        });
        !cancel
    }

    pub(in crate::game) fn mod_slot_changed(
        &mut self,
        holder: EntityRef,
        slot: &str,
        index: usize,
        cosmetic: bool,
        old: Option<mod_api::ItemStack>,
        new: Option<mod_api::ItemStack>,
        source: &str,
    ) {
        self.native_mod_slot_changed(mod_api::__private::ModSlotChangedCall {
            holder,
            slot,
            index,
            cosmetic,
            old: old.as_ref(),
            new: new.as_ref(),
            source,
        });
    }
}
