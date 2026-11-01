module
public import GemmProofs.Core.Model

@[expose] public section
namespace GemmProofs.Core

/-- One max-plus recurrence step, including the source base case. -/
def advance (G : Graph V) (upper : Bool) (τ : Trace V) (v : V) : Nat :=
  if (G.preds v).Nonempty then
    (G.preds v).sup (fun e => τ e.src + if upper then e.hi else e.lo)
  else if upper then G.startHi v else G.startLo v

/-- Recursion is justified by the supplied rank, not by assuming a schedule exists. -/
def schedule (G : Graph V) (r : V → Nat) (hr : G.IsRankedDAG r)
    (upper : Bool) (v : V) : Nat :=
  if (G.preds v).Nonempty then
    (G.preds v).attach.sup (fun e =>
      schedule G r hr upper e.val.src + if upper then e.val.hi else e.val.lo)
  else if upper then G.startHi v else G.startLo v
termination_by r v
 decreasing_by exact hr v e.val e.property

theorem schedule_eq (G : Graph V) (r : V → Nat) (hr : G.IsRankedDAG r)
    (upper : Bool) (v : V) :
    schedule G r hr upper v = advance G upper (schedule G r hr upper) v := by
  rw [schedule]
  unfold advance
  by_cases hp : (G.preds v).Nonempty
  · simp only [ite_eq_left hp]
    exact Finset.sup_attach (G.preds v)
      (fun e => schedule G r hr upper e.src + if upper then e.hi else e.lo)
  · simp only [ite_eq_right hp]

theorem lower_iff (G : Graph V) (τ : Trace V) :
    G.ValidLower τ ↔ ∀ v, advance G false τ v ≤ τ v := by
  constructor
  · intro h v
    simp only [advance, Bool.false_eq_true, ↓reduceIte]
    split_ifs with hp
    · exact Finset.sup_le (fun e he => h.2 v e he)
    · exact h.1 v (Finset.not_nonempty_iff_eq_empty.mp hp)
  · intro h
    constructor
    · intro v hv
      simpa [advance, hv] using h v
    · intro v e he
      have hp : (G.preds v).Nonempty := ⟨e, he⟩
      exact (Finset.le_sup he).trans (by simpa [advance, hp] using h v)

theorem upper_iff (G : Graph V) (τ : Trace V) :
    G.ValidUpper τ ↔ ∀ v, τ v ≤ advance G true τ v := by
  simp only [Graph.ValidUpper, advance, ↓reduceIte]
  constructor <;> intro h v <;> specialize h v
  all_goals split_ifs at * <;> assumption

theorem advance_lo_le_hi (G : Graph V) (hw : G.WellFormed) (τ : Trace V) (v : V) :
    advance G false τ v ≤ advance G true τ v := by
  simp only [advance, Bool.false_eq_true, ↓reduceIte]
  split_ifs with hp
  · apply Finset.sup_mono_fun
    intro e he
    exact Nat.add_le_add_left (hw.2 v e he) _
  · exact hw.1 v (Finset.not_nonempty_iff_eq_empty.mp hp)

theorem schedule_le (G : Graph V) (r : V → Nat) (hr : G.IsRankedDAG r)
    (upper : Bool) (τ : Trace V) (hτ : ∀ v, advance G upper τ v ≤ τ v) :
    ∀ v, schedule G r hr upper v ≤ τ v := by
  intro v
  induction v using (measure r).wf.induction with
  | h v ih =>
    rw [schedule_eq]
    apply Nat.le_trans _ (hτ v)
    by_cases hp : (G.preds v).Nonempty
    · simp only [advance, ite_eq_left hp]
      apply Finset.sup_mono_fun
      intro e he
      exact Nat.add_le_add_right (ih e.src (hr v e he)) _
    · simp only [advance, ite_eq_right hp, Nat.le_refl]

theorem le_schedule (G : Graph V) (r : V → Nat) (hr : G.IsRankedDAG r)
    (upper : Bool) (τ : Trace V) (hτ : ∀ v, τ v ≤ advance G upper τ v) :
    ∀ v, τ v ≤ schedule G r hr upper v := by
  intro v
  induction v using (measure r).wf.induction with
  | h v ih =>
    rw [schedule_eq]
    apply Nat.le_trans (hτ v)
    by_cases hp : (G.preds v).Nonempty
    · simp only [advance, ite_eq_left hp]
      apply Finset.sup_mono_fun
      intro e he
      exact Nat.add_le_add_right (ih e.src (hr v e he)) _
    · simp only [advance, ite_eq_right hp, Nat.le_refl]

theorem lower_schedule_valid (G : Graph V) (hw : G.WellFormed)
    (r : V → Nat) (hr : G.IsRankedDAG r) :
    G.ValidLower (schedule G r hr false) ∧ G.ValidUpper (schedule G r hr false) := by
  constructor
  · apply (lower_iff G _).mpr
    intro v
    rw [← schedule_eq]
  · apply (upper_iff G _).mpr
    intro v
    calc
      schedule G r hr false v = advance G false (schedule G r hr false) v := schedule_eq ..
      _ ≤ advance G true (schedule G r hr false) v := advance_lo_le_hi G hw _ v

theorem upper_schedule_valid (G : Graph V) (hw : G.WellFormed)
    (r : V → Nat) (hr : G.IsRankedDAG r) :
    G.ValidLower (schedule G r hr true) ∧ G.ValidUpper (schedule G r hr true) := by
  constructor
  · apply (lower_iff G _).mpr
    intro v
    calc
      advance G false (schedule G r hr true) v ≤ advance G true (schedule G r hr true) v :=
        advance_lo_le_hi G hw _ v
      _ = schedule G r hr true v := (schedule_eq ..).symm
  · apply (upper_iff G _).mpr
    intro v
    rw [← schedule_eq]

theorem extrema (V : Type) (G : Graph V) (hw : G.WellFormed)
    (r : V → Nat) (hr : G.IsRankedDAG r) :
    ∃ low high : Trace V,
      G.ValidLower low ∧ G.ValidUpper low ∧
      G.ValidLower high ∧ G.ValidUpper high ∧
      (∀ v, low v = advance G false low v) ∧
      (∀ v, high v = advance G true high v) ∧
      (∀ τ, G.ValidLower τ → ∀ v, low v ≤ τ v) ∧
      (∀ τ, G.ValidUpper τ → ∀ v, τ v ≤ high v) := by
  refine ⟨schedule G r hr false, schedule G r hr true,
    (lower_schedule_valid G hw r hr).1, (lower_schedule_valid G hw r hr).2,
    (upper_schedule_valid G hw r hr).1, (upper_schedule_valid G hw r hr).2,
    schedule_eq G r hr false, schedule_eq G r hr true, ?_, ?_⟩
  · intro τ hτ
    exact schedule_le G r hr false τ ((lower_iff G τ).mp hτ)
  · intro τ hτ
    exact le_schedule G r hr true τ ((upper_iff G τ).mp hτ)

theorem finish_lower (G : Graph V) (r : V → Nat) (hr : G.IsRankedDAG r)
    (O : Finset V) (τ : Trace V) (hτ : G.ValidLower τ) :
    finish O (schedule G r hr false) ≤ finish O τ :=
  finish_mono O (schedule_le G r hr false τ ((lower_iff G τ).mp hτ))

theorem finish_upper (G : Graph V) (r : V → Nat) (hr : G.IsRankedDAG r)
    (O : Finset V) (τ : Trace V) (hτ : G.ValidUpper τ) :
    finish O τ ≤ finish O (schedule G r hr true) :=
  finish_mono O (le_schedule G r hr true τ ((upper_iff G τ).mp hτ))

theorem observation_incompatible (G : Graph V) (r : V → Nat) (hr : G.IsRankedDAG r)
    (O : Finset V) (I : Set Nat)
    (sep : ∀ t ∈ I, t < finish O (schedule G r hr false) ∨
      finish O (schedule G r hr true) < t) :
    ¬ ∃ τ, G.ValidLower τ ∧ G.ValidUpper τ ∧ finish O τ ∈ I := by
  rintro ⟨τ, hlo, hhi, ht⟩
  have lo := finish_lower G r hr O τ hlo
  have hi := finish_upper G r hr O τ hhi
  rcases sep _ ht with h | h <;> omega

theorem advance_fixed (G : Graph V)
    (hs : ∀ v, G.preds v = ∅ → G.startLo v = G.startHi v)
    (he : ∀ v e, e ∈ G.preds v → e.lo = e.hi) (τ : Trace V) (v : V) :
    advance G false τ v = advance G true τ v := by
  simp only [advance, Bool.false_eq_true, ↓reduceIte]
  split_ifs with hp
  · apply Finset.sup_congr rfl
    intro e hm
    rw [he v e hm]
  · exact hs v (Finset.not_nonempty_iff_eq_empty.mp hp)

theorem unique_schedule (V : Type) (G : Graph V) (hw : G.WellFormed)
    (hs : ∀ v, G.preds v = ∅ → G.startLo v = G.startHi v)
    (he : ∀ v e, e ∈ G.preds v → e.lo = e.hi)
    (r : V → Nat) (hr : G.IsRankedDAG r) :
    ∃! τ, G.ValidLower τ ∧ G.ValidUpper τ := by
  refine ⟨schedule G r hr false, lower_schedule_valid G hw r hr, ?_⟩
  intro τ hτ
  funext v
  apply Nat.le_antisymm
  · apply le_schedule G r hr false τ _ v
    intro w
    rw [advance_fixed G hs he]
    exact (upper_iff G τ).mp hτ.2 w
  · exact schedule_le G r hr false τ ((lower_iff G τ).mp hτ.1) v

theorem schedule_order (G : Graph V) (hw : G.WellFormed)
    (r : V → Nat) (hr : G.IsRankedDAG r) (v : V) :
    schedule G r hr false v ≤ schedule G r hr true v :=
  schedule_le G r hr false _
    ((lower_iff G _).mp (upper_schedule_valid G hw r hr).1) v

theorem fixed_schedule_eq (G : Graph V) (hw : G.WellFormed)
    (hs : ∀ v, G.preds v = ∅ → G.startLo v = G.startHi v)
    (he : ∀ v e, e ∈ G.preds v → e.lo = e.hi)
    (r : V → Nat) (hr : G.IsRankedDAG r) :
    schedule G r hr false = schedule G r hr true := by
  obtain ⟨τ, _, unique⟩ := unique_schedule V G hw hs he r hr
  exact (unique _ (lower_schedule_valid G hw r hr)).trans
    (unique _ (upper_schedule_valid G hw r hr)).symm

end GemmProofs.Core
