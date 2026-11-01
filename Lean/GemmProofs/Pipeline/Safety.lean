module
public import GemmProofs.Pipeline.Model
public import Mathlib.Data.Nat.ModEq
public import Mathlib.Data.Finset.Card
public import Mathlib.Order.Interval.Finset.Nat
public import Mathlib.Algebra.Order.BigOperators.Group.Finset

@[expose] public section
namespace GemmProofs.Pipeline
open Core
open scoped BigOperators

theorem lifecycle {n S : Nat} {p : Params} {cross : Bool}
    {τ : Trace (Node n)} (hτ : (graph n S p cross).ValidLower τ) (g : Fin n) :
    τ (g, .acq) + p.load.lo ≤ τ (g, .rdy) ∧
    τ (g, .rdy) ≤ τ (g, .commit) ∧
    τ (g, .commit) + p.mma.lo ≤ τ (g, .cmp) ∧
    τ (g, .cmp) ≤ τ (g, .rel) := by
  have a := hτ.2 (g, .rdy) (incoming g .acq p.load) (by simp [graph])
  have b := hτ.2 (g, .commit) (incoming g .rdy zero) (by simp [graph])
  have c := hτ.2 (g, .cmp) (incoming g .commit p.mma) (by simp [graph])
  have d := hτ.2 (g, .rel) (incoming g .cmp zero) (by simp [graph])
  simp only [incoming, zero, Nat.add_zero] at a b c d
  exact ⟨a, b, c, d⟩

theorem minimum_occupancy {n S : Nat} {p : Params} {cross : Bool}
    {τ : Trace (Node n)} (hτ : (graph n S p cross).ValidLower τ) (g : Fin n) :
    τ (g, .acq) + (p.load.lo + p.mma.lo) ≤ τ (g, .rel) := by
  obtain ⟨a, b, c, d⟩ := lifecycle hτ g
  omega

theorem reuse {n S : Nat} {p : Params} {cross : Bool}
    {τ : Trace (Node n)} (hτ : (graph n S p cross).ValidLower τ)
    (g h : Fin n) (heq : h.val = g.val + S) : τ (g, .rel) ≤ τ (h, .acq) := by
  have hm : incoming g .rel zero ∈ (graph n S p cross).preds (h, .acq) := by
    simp [graph, heq, incoming]
  simpa [incoming, zero] using hτ.2 (h, .acq) (incoming g .rel zero) hm

theorem same_slot_order {n S : Nat} {p : Params} {cross : Bool} (hs : 1 ≤ S)
    {τ : Trace (Node n)} (hτ : (graph n S p cross).ValidLower τ)
    (g h : Fin n) (hgh : g.val < h.val) (hmod : g.val % S = h.val % S) :
    τ (g, .rel) ≤ τ (h, .acq) := by
  induction h using (measure (fun h : Fin n => h.val)).wf.induction with
  | h h ih =>
    have hdiv : S ∣ h.val - g.val := (Nat.modEq_iff_dvd' (Nat.le_of_lt hgh)).mp hmod
    have hgap : S ≤ h.val - g.val := Nat.le_of_dvd (by omega) hdiv
    let k : Fin n := ⟨h.val - S, by omega⟩
    have hk : k.val < h.val := by dsimp [k]; omega
    have heq : h.val = k.val + S := by dsimp [k]; omega
    have hkmod : g.val % S = k.val % S := hmod.trans (Nat.mod_eq_sub_mod (by omega))
    have hkg : g.val ≤ k.val := by dsimp [k]; omega
    by_cases he : g = k
    · rw [he]
      exact reuse hτ k h heq
    · have hlt : g.val < k.val := by
        have hn : g.val ≠ k.val := fun h => he (Fin.ext h)
        omega
      have a := ih k hk hlt hkmod
      have b := minimum_occupancy hτ k
      have c := reuse hτ k h heq
      omega

theorem slot_disjoint (n S : Nat) (p : Params) (cross : Bool) (hs : 1 ≤ S)
    (τ : Trace (Node n)) (hτ : (graph n S p cross).ValidLower τ)
    (g h : Fin n) (hgh : g.val < h.val) (hmod : g.val % S = h.val % S) :
    τ (g, .cmp) ≤ τ (g, .rel) ∧ τ (g, .rel) ≤ τ (h, .acq) :=
  ⟨(lifecycle hτ g).2.2.2, same_slot_order hs hτ g h hgh hmod⟩

theorem capacity (n S : Nat) (p : Params) (cross : Bool) (hs : 1 ≤ S)
    (τ : Trace (Node n)) (hτ : (graph n S p cross).ValidLower τ) (t : Nat) :
    (Finset.univ.filter (fun g : Fin n => τ (g, .acq) ≤ t ∧ t < τ (g, .rel))).card ≤ S := by
  let active := Finset.univ.filter (fun g : Fin n => τ (g, .acq) ≤ t ∧ t < τ (g, .rel))
  have bound : active.card ≤ (Finset.range S).card := by
    apply Finset.card_le_card_of_injOn (fun g : Fin n => g.val % S)
    · intro g _
      exact Finset.mem_range.mpr (Nat.mod_lt _ (by omega))
    · intro g hg h hh heq
      have ag := (Finset.mem_filter.mp hg).2
      have ah := (Finset.mem_filter.mp hh).2
      apply Fin.ext
      rcases Nat.lt_trichotomy g.val h.val with hlt | he | hgt
      · have := same_slot_order hs hτ g h hlt heq
        omega
      · exact he
      · have := same_slot_order hs hτ h g hgt heq.symm
        omega
  simpa [active] using bound

theorem release_le_finish (τ : Trace (Node n)) (g : Fin n) :
    τ (g, .rel) ≤ finish (observations n) τ :=
  le_finish _ _ (by simp [observations])

theorem event_le_finish {n S : Nat} {p : Params} {cross : Bool}
    {τ : Trace (Node n)} (hτ : (graph n S p cross).ValidLower τ) (v : Node n) :
    τ v ≤ finish (observations n) τ := by
  obtain ⟨a, b, c, d⟩ := lifecycle hτ v.1
  have hu := release_le_finish τ v.1
  rcases v with ⟨g, k⟩
  cases k <;> dsimp at * <;> omega

/-- The selected release observations measure completion of all five-event items. -/
theorem finish_all_events {n S : Nat} {p : Params} {cross : Bool}
    {τ : Trace (Node n)} (hτ : (graph n S p cross).ValidLower τ) :
    finish (observations n) τ = finish Finset.univ τ := by
  apply Nat.le_antisymm
  · apply Finset.sup_mono
    exact Finset.subset_univ _
  · apply Finset.sup_le
    intro v _
    exact event_le_finish hτ v

/-- Count the same occupied (item, time) pairs by item and by time. The per-time
bound is the injective slot map proved above, on this very same trace. -/
theorem occupancy (n S : Nat) (p : Params) (cross : Bool) (hs : 1 ≤ S)
    (τ : Trace (Node n)) (hτ : (graph n S p cross).ValidLower τ) :
    n * (p.load.lo + p.mma.lo) ≤ S * finish (observations n) τ := by
  let T := finish (observations n) τ
  have interval (g : Fin n) :
      (Finset.range T).filter (fun t => τ (g, .acq) ≤ t ∧ t < τ (g, .rel)) =
        Finset.Ico (τ (g, .acq)) (τ (g, .rel)) := by
    ext t
    have hu := release_le_finish τ g
    simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_Ico]
    dsimp [T]
    omega
  have count :
      (∑ g : Fin n, (τ (g, .rel) - τ (g, .acq))) =
      ∑ t ∈ Finset.range T,
        (Finset.univ.filter (fun g : Fin n => τ (g, .acq) ≤ t ∧ t < τ (g, .rel))).card := by
    calc
      _ = ∑ g : Fin n,
          ((Finset.range T).filter (fun t => τ (g, .acq) ≤ t ∧ t < τ (g, .rel))).card := by
        apply Finset.sum_congr rfl
        intro g _
        rw [interval g, Nat.card_Ico]
      _ = ∑ g : Fin n, ∑ t ∈ Finset.range T,
          if τ (g, .acq) ≤ t ∧ t < τ (g, .rel) then 1 else 0 := by
        simp only [Finset.card_eq_sum_ones, Finset.sum_filter]
      _ = ∑ t ∈ Finset.range T, ∑ g : Fin n,
          if τ (g, .acq) ≤ t ∧ t < τ (g, .rel) then 1 else 0 := Finset.sum_comm
      _ = _ := by simp only [Finset.card_eq_sum_ones, Finset.sum_filter]
  calc
    n * (p.load.lo + p.mma.lo) = ∑ _g : Fin n, (p.load.lo + p.mma.lo) := by simp
    _ ≤ ∑ g : Fin n, (τ (g, .rel) - τ (g, .acq)) := by
      apply Finset.sum_le_sum
      intro g _
      have := minimum_occupancy hτ g
      omega
    _ = _ := count
    _ ≤ ∑ _t ∈ Finset.range T, S := by
      apply Finset.sum_le_sum
      intro t _
      exact capacity n S p cross hs τ hτ t
    _ = S * finish (observations n) τ := by simp [T, Nat.mul_comm]

end GemmProofs.Pipeline
