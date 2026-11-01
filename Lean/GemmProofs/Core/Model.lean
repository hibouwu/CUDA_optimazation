module
public import Mathlib.Data.Finset.Lattice.Fold
public import Mathlib.Data.Fintype.Prod
public import Mathlib.Tactic.SplitIfs
public import Lean.Elab.Tactic.Omega

@[expose] public section

/-! The interval graph is independent of a rank certificate and observation set.
Incoming edges are a finite set; duplicate identical constraints have no effect. -/
namespace GemmProofs.Core

structure Incoming (V : Type) where
  src : V
  lo : Nat
  hi : Nat
  deriving DecidableEq

structure Graph (V : Type) where
  preds : V → Finset (Incoming V)
  startLo : V → Nat
  startHi : V → Nat

abbrev Trace (V : Type) := V → Nat

def finish (O : Finset V) (τ : Trace V) : Nat := O.sup τ

def Graph.WellFormed (G : Graph V) : Prop :=
  (∀ v, G.preds v = ∅ → G.startLo v ≤ G.startHi v) ∧
  (∀ v e, e ∈ G.preds v → e.lo ≤ e.hi)

def Graph.ValidLower (G : Graph V) (τ : Trace V) : Prop :=
  (∀ v, G.preds v = ∅ → G.startLo v ≤ τ v) ∧
  (∀ v e, e ∈ G.preds v → τ e.src + e.lo ≤ τ v)

def Graph.ValidUpper (G : Graph V) (τ : Trace V) : Prop :=
  ∀ v, if (G.preds v).Nonempty then τ v ≤ (G.preds v).sup (fun e => τ e.src + e.hi)
    else τ v ≤ G.startHi v

def Graph.IsRankedDAG (G : Graph V) (rank : V → Nat) : Prop :=
  ∀ v e, e ∈ G.preds v → rank e.src < rank v

@[simp] theorem finish_empty (τ : Trace V) : finish ∅ τ = 0 := by
  simp [finish]

theorem le_finish (O : Finset V) (τ : Trace V) {v : V} (hv : v ∈ O) :
    τ v ≤ finish O τ := Finset.le_sup hv

theorem finish_mono (O : Finset V) {τ σ : Trace V} (h : ∀ v, τ v ≤ σ v) :
    finish O τ ≤ finish O σ := by
  apply Finset.sup_le
  intro v hv
  exact (h v).trans (le_finish O σ hv)

end GemmProofs.Core
