module
public import GemmProofs

@[expose] public section

/-! Delivery contracts, fixed before their proof implementations. Each contract
below is bound to a library theorem; axiom dependencies are checked by check.sh. -/
open GemmProofs GemmProofs.Core GemmProofs.Pipeline

namespace Contract

abbrev ExtremalSchedules : Prop :=
  ∀ (V : Type) (G : Graph V), G.WellFormed →
  ∀ r : V → Nat, G.IsRankedDAG r →
  ∃ low high : Trace V,
    G.ValidLower low ∧ G.ValidUpper low ∧
    G.ValidLower high ∧ G.ValidUpper high ∧
    (∀ v, low v = if (G.preds v).Nonempty then
      (G.preds v).sup (fun e => low e.src + e.lo) else G.startLo v) ∧
    (∀ v, high v = if (G.preds v).Nonempty then
      (G.preds v).sup (fun e => high e.src + e.hi) else G.startHi v) ∧
    (∀ τ, G.ValidLower τ → ∀ v, low v ≤ τ v) ∧
    (∀ τ, G.ValidUpper τ → ∀ v, τ v ≤ high v)

abbrev UniqueSchedule : Prop :=
  ∀ (V : Type) (G : Graph V), G.WellFormed →
  (∀ v, G.preds v = ∅ → G.startLo v = G.startHi v) →
  (∀ v e, e ∈ G.preds v → e.lo = e.hi) →
  ∀ r : V → Nat, G.IsRankedDAG r →
  ∃! τ, G.ValidLower τ ∧ G.ValidUpper τ

abbrev SlotDisjoint : Prop :=
  ∀ n S (p : Params) (cross : Bool), 1 ≤ S →
  ∀ τ, (graph n S p cross).ValidLower τ →
  ∀ g h : Fin n, g.val < h.val → g.val % S = h.val % S →
    τ (g, .cmp) ≤ τ (g, .rel) ∧ τ (g, .rel) ≤ τ (h, .acq)

abbrev Capacity : Prop :=
  ∀ n S (p : Params) (cross : Bool), 1 ≤ S →
  ∀ τ, (graph n S p cross).ValidLower τ → ∀ t : Nat,
    (Finset.univ.filter (fun g : Fin n => τ (g, .acq) ≤ t ∧ t < τ (g, .rel))).card ≤ S

abbrev Occupancy : Prop :=
  ∀ n S (p : Params) (cross : Bool), 1 ≤ S →
  ∀ τ, (graph n S p cross).ValidLower τ →
    n * (p.load.lo + p.mma.lo) ≤ S * finish (observations n) τ

abbrev SingleSlotImpossible : Prop :=
  ∀ n (p : Params), 2 ≤ n → 0 < p.load.lo →
    ¬ ∃ τ, (graph n 1 p true).ValidLower τ

abbrev FixedExample (S : Nat) (cross : Bool) (T : Nat) : Prop :=
  let G := graph 12 S (fixed 1 4 2) cross
  (∃ τ, G.ValidLower τ ∧ G.ValidUpper τ ∧ finish (observations 12) τ = T) ∧
  (∀ τ, G.ValidLower τ → T ≤ finish (observations 12) τ) ∧
  (∀ τ, G.ValidUpper τ → finish (observations 12) τ ≤ T)

end Contract

example (n S : Nat) (p : Params) (cross : Bool) :
    2 ≤ S → (graph n S p cross).IsRankedDAG rank := ranked n S p cross

example (n S : Nat) (p : Params) (cross : Bool) :
    (graph n S p cross).WellFormed := wellFormed n S p cross

example (n S : Nat) (p : Params) (cross : Bool) (hS : 1 ≤ S) (v : Node n) :
    (graph n S p cross).preds v = ∅ ↔ v.1.val = 0 ∧ v.2 = .acq :=
  source_iff n S p cross hS v

example (τ : Trace (Node 0)) : finish (observations 0) τ = 0 := empty_finish τ

#print axioms GemmProofs.Core.finish_empty
#print axioms GemmProofs.Core.le_finish
#print axioms GemmProofs.Core.finish_mono
#print axioms GemmProofs.Pipeline.ranked
#print axioms GemmProofs.Pipeline.wellFormed
#print axioms GemmProofs.Pipeline.source_iff
#print axioms GemmProofs.Pipeline.empty_finish

example (S : Nat) (p : Params) (cross : Bool) (τ : Trace (Node 0)) :
    (graph 0 S p cross).ValidLower τ ∧ (graph 0 S p cross).ValidUpper τ :=
  empty_valid S p cross τ

example (τ σ : Trace (Node 0)) : τ = σ := empty_unique τ σ

#print axioms GemmProofs.Pipeline.empty_valid
#print axioms GemmProofs.Pipeline.empty_unique

example : Contract.ExtremalSchedules := Core.extrema
example : Contract.UniqueSchedule := Core.unique_schedule

example {V : Type} {G : Graph V} {τ : Trace V} (hτ : G.ValidLower τ)
    {u v : V} {d : Nat} (path : LowerPath G u v d) : τ u + d ≤ τ v :=
  path_lower hτ path

example {V : Type} {G : Graph V} {v : V} {d : Nat}
    (path : LowerPath G v v d) (hd : 0 < d) : ¬ ∃ τ, G.ValidLower τ :=
  positive_cycle_impossible path hd

#print axioms GemmProofs.Core.path_lower
#print axioms GemmProofs.Core.source_path_finish
#print axioms GemmProofs.Core.positive_cycle_impossible
#print axioms GemmProofs.Core.covered_le_finish
#print axioms GemmProofs.Core.extrema
#print axioms GemmProofs.Core.unique_schedule
#print axioms GemmProofs.Core.finish_lower
#print axioms GemmProofs.Core.finish_upper
#print axioms GemmProofs.Core.observation_incompatible

example : Contract.SlotDisjoint := Pipeline.slot_disjoint
example : Contract.Capacity := Pipeline.capacity
example : Contract.Occupancy := Pipeline.occupancy

#print axioms GemmProofs.Pipeline.minimum_occupancy
#print axioms GemmProofs.Pipeline.slot_disjoint
#print axioms GemmProofs.Pipeline.capacity
#print axioms GemmProofs.Pipeline.occupancy

example {V : Type} (G : Graph V) (r : V → Nat) (hr : G.IsRankedDAG r)
    (O : Finset V) (I : Set Nat)
    (sep : ∀ t ∈ I, t < finish O (schedule G r hr false) ∨
      finish O (schedule G r hr true) < t) :
    ¬ ∃ τ, G.ValidLower τ ∧ G.ValidUpper τ ∧ finish O τ ∈ I :=
  observation_incompatible G r hr O I sep

#print axioms GemmProofs.Core.schedule_order
#print axioms GemmProofs.Core.fixed_schedule_eq

example : Contract.SingleSlotImpossible := Pipeline.single_slot_impossible
example : Contract.FixedExample 4 true 22 := Pipeline.four_slot_tight
example : Contract.FixedExample 4 false 21 := Pipeline.four_slot_no_cross_tight
example : Contract.FixedExample 6 true 17 := Pipeline.six_slot_tight

example : LowerPath (graph 12 4 (fixed 1 4 2) true) (0, .acq) (11, .rel) 22 :=
  critical_path22
example : LowerPath (graph 12 4 (fixed 1 4 2) false) (0, .acq) (11, .rel) 21 :=
  critical_path21
example : LowerPath (graph 12 6 (fixed 1 4 2) true) (0, .acq) (11, .rel) 17 :=
  critical_path17

#print axioms GemmProofs.Pipeline.single_slot_impossible
#print axioms GemmProofs.Pipeline.witness4_valid
#print axioms GemmProofs.Pipeline.witness4NoCross_valid
#print axioms GemmProofs.Pipeline.witness6_valid
#print axioms GemmProofs.Pipeline.four_slot_tight
#print axioms GemmProofs.Pipeline.four_slot_no_cross_tight
#print axioms GemmProofs.Pipeline.six_slot_tight
#print axioms GemmProofs.Pipeline.critical_path22
#print axioms GemmProofs.Pipeline.critical_path21
#print axioms GemmProofs.Pipeline.critical_path17
#print axioms GemmProofs.Pipeline.slot_comparison
#print axioms GemmProofs.Pipeline.cross_edge_comparison

example : Contract.FixedExample 6 false 17 := Pipeline.six_slot_no_cross_tight

example {n S : Nat} {p : Params} {cross : Bool} {τ : Trace (Node n)}
    (hτ : (graph n S p cross).ValidLower τ) :
    finish (observations n) τ = finish Finset.univ τ := finish_all_events hτ

#print axioms GemmProofs.Pipeline.finish_all_events
#print axioms GemmProofs.Pipeline.six_slot_no_cross_tight

-- Computation-service contracts extend, rather than replace, the stage-one inventory.
namespace Contract

abbrev ComputeResidual : Prop :=
  ∀ M Eprev c q lag f : Nat, f ≤ c →
    let E := max M Eprev + c
    let C := max (M + f) (E - q)
    let D := E + lag
    lag ≤ D - C ∧ D - C ≤ q + lag

abbrev ComputeWindow : Prop :=
  ∀ M Eprev c q lag f : Nat, f ≤ c →
    let E := max M Eprev + c
    let C := max (M + f) (E - q)
    let D := E + lag
    C + lag ≤ D ∧ D ≤ C + (q + lag)

end Contract

example : Contract.ComputeResidual := Compute.residual_bounds
example : Contract.ComputeWindow := Compute.completion_window

example (M Eprev c q lag f : Int) (hfc : f ≤ c) (hq : 0 ≤ q) :
    let E := max M Eprev + c
    let C := max (M + f) (E - q)
    let D := E + lag
    C + lag ≤ D ∧ D ≤ C + (q + lag) :=
  Compute.signed_completion_window M Eprev c q lag f hfc hq

example (M Eprev c q f : Nat) :
    (Compute.commit M Eprev c q f : Int) =
      max ((M : Int) + f) (max (M : Int) Eprev + c - q) := Compute.commit_cast ..

example (M Eprev c q lag f lo hi qMax : Nat)
    (hfc : f ≤ c) (hlo : lo ≤ lag) (hhi : lag ≤ hi) (hq : q ≤ qMax) :
    lo ≤ qMax + hi ∧
    Compute.commit M Eprev c q f + lo ≤ Compute.complete M Eprev c lag ∧
    Compute.complete M Eprev c lag ≤ Compute.commit M Eprev c q f + (qMax + hi) :=
  Compute.edge_bounds M Eprev c q lag f lo hi qMax hfc hlo hhi hq

example (G : Graph V) (τ : Trace V) (u v : V)
    (M Eprev c q lag f lo hi qMax : Nat)
    (hfc : f ≤ c) (hlo : lo ≤ lag) (hhi : lag ≤ hi) (hq : q ≤ qMax)
    (hu : τ u = Compute.commit M Eprev c q f)
    (hv : τ v = Compute.complete M Eprev c lag)
    (he : (⟨u, lo, qMax + hi⟩ : Incoming V) ∈ G.preds v) :
    τ u + lo ≤ τ v ∧ τ v ≤ (G.preds v).sup (fun e => τ e.src + e.hi) :=
  Compute.graph_edge G τ u v M Eprev c q lag f lo hi qMax hfc hlo hhi hq hu hv he

-- Directed boundary witnesses: both maxima, large q, zero delay, and necessity of f ≤ c.
example : Compute.complete 0 0 10 1 - Compute.commit 0 0 10 3 0 = 4 := by decide
example : Compute.complete 2 30 10 1 - Compute.commit 2 30 10 3 0 = 4 := by decide
example : Compute.complete 10 0 10 2 - Compute.commit 10 0 10 12 9 = 3 := by decide
example : Compute.commit 0 0 10 20 0 = 0 ∧ Compute.complete 0 0 10 1 = 11 := by decide
example : Compute.complete 0 0 0 0 - Compute.commit 0 0 0 0 0 = 0 := by decide
example : ¬ (1 ≤ Compute.complete 0 0 1 1 - Compute.commit 0 0 1 0 3) := by decide

#print axioms GemmProofs.Pipeline.Compute.signed_completion_window
#print axioms GemmProofs.Pipeline.Compute.commit_cast
#print axioms GemmProofs.Pipeline.Compute.completion_window
#print axioms GemmProofs.Pipeline.Compute.residual_bounds
#print axioms GemmProofs.Pipeline.Compute.edge_bounds
#print axioms GemmProofs.Pipeline.Compute.graph_edge
