module
public import GemmProofs.Core.Model

@[expose] public section

namespace GemmProofs.Pipeline
open Core

inductive Event where
  | acq | rdy | commit | cmp | rel
  deriving DecidableEq

instance : Fintype Event where
  elems := {.acq, .rdy, .commit, .cmp, .rel}
  complete := by intro k; cases k <;> simp

abbrev Node (n : Nat) := Fin n × Event

structure Delay where
  lo : Nat
  hi : Nat
  ordered : lo ≤ hi
  deriving DecidableEq

structure Params where
  issue : Delay
  load : Delay
  mma : Delay
  deriving DecidableEq

def fixed (issue load mma : Nat) : Params :=
  ⟨⟨issue, issue, Nat.le_refl _⟩, ⟨load, load, Nat.le_refl _⟩,
    ⟨mma, mma, Nat.le_refl _⟩⟩

def incoming (g : Fin n) (e : Event) (d : Delay) : Incoming (Node n) :=
  ⟨(g, e), d.lo, d.hi⟩

def zero : Delay := ⟨0, 0, Nat.le_refl _⟩

/-- `cross = false` removes precisely the cross-iteration commit-to-release edges. -/
def graph (n S : Nat) (p : Params) (cross : Bool := true) : Graph (Node n) where
  startLo := fun _ => 0
  startHi := fun _ => 0
  preds := fun (g, e) => match e with
    | .acq =>
      (if h : 0 < g.val then
        {incoming ⟨g.val - 1, by omega⟩ .acq p.issue} else ∅) ∪
      (if h : S ≤ g.val then
        {incoming ⟨g.val - S, by omega⟩ .rel zero} else ∅)
    | .rdy => {incoming g .acq p.load}
    | .commit => {incoming g .rdy zero}
    | .cmp => {incoming g .commit p.mma}
    | .rel => {incoming g .cmp zero} ∪
      (if h : g.val + 1 < n then
        if cross then {incoming ⟨g.val + 1, h⟩ .commit zero} else ∅
       else ∅)

def observations (n : Nat) : Finset (Node n) :=
  Finset.univ.filter (fun v => v.2 = .rel)

def rank (v : Node n) : Nat := 5 * v.1.val + match v.2 with
  | .acq => 0 | .rdy => 1 | .commit => 2 | .cmp => 3 | .rel => 8

/-- Every local edge of ARCH §3.1 increases this rank. -/
theorem ranked (n S : Nat) (p : Params) (cross : Bool) (hS : 2 ≤ S) :
    (graph n S p cross).IsRankedDAG rank := by
  intro ⟨g, k⟩ e he
  cases k <;> simp only [graph] at he
  all_goals try split_ifs at he
  all_goals simp only [Finset.mem_union, Finset.mem_singleton,
    Finset.notMem_empty, false_or, or_false] at he
  all_goals rcases he with rfl | rfl
  all_goals dsimp [incoming, rank]; omega

theorem wellFormed (n S : Nat) (p : Params) (cross : Bool) :
    (graph n S p cross).WellFormed := by
  constructor
  · intro v hv
    exact Nat.le_refl _
  · intro ⟨g, k⟩ e he
    cases k <;> simp only [graph] at he
    all_goals try split_ifs at he
    all_goals simp only [Finset.mem_union, Finset.mem_singleton,
      Finset.notMem_empty, false_or, or_false] at he
    all_goals rcases he with rfl | rfl
    all_goals dsimp [incoming, zero]
    all_goals first | exact p.issue.ordered | exact p.load.ordered |
      exact p.mma.ordered | exact Nat.le_refl _

/-- Under the specified positive slot domain, the only source is acq(0). -/
theorem source_iff (n S : Nat) (p : Params) (cross : Bool) (hS : 1 ≤ S)
    (v : Node n) :
    (graph n S p cross).preds v = ∅ ↔ v.1.val = 0 ∧ v.2 = .acq := by
  rcases v with ⟨g, k⟩
  cases k <;> simp only [graph]
  all_goals try split_ifs
  all_goals simp_all
  all_goals omega

@[simp] theorem observations_zero : observations 0 = ∅ := by
  ext v
  exact Fin.elim0 v.1

theorem empty_finish (τ : Trace (Node 0)) : finish (observations 0) τ = 0 := by
  simp

theorem empty_valid (S : Nat) (p : Params) (cross : Bool) (τ : Trace (Node 0)) :
    (graph 0 S p cross).ValidLower τ ∧ (graph 0 S p cross).ValidUpper τ := by
  constructor
  · constructor
    · intro v; exact Fin.elim0 v.1
    · intro v; exact Fin.elim0 v.1
  · intro v; exact Fin.elim0 v.1

theorem empty_unique (τ σ : Trace (Node 0)) : τ = σ := by
  funext v
  exact Fin.elim0 v.1

end GemmProofs.Pipeline
