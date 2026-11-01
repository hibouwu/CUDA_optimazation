module
public import GemmProofs.Core.Model

@[expose] public section
namespace GemmProofs.Pipeline.Compute
open Core

/-! The E/C/D part of `wgmma_compute_step`, in the graph's integer time units.
No wait-return rule, floating-point equivalence, or hardware timing claim is made.
The signed lemma and cast identity check that Nat subtraction does not change C. -/

def serviceEnd (M Eprev c : Nat) : Nat := max M Eprev + c

def commit (M Eprev c q f : Nat) : Nat :=
  max (M + f) (serviceEnd M Eprev c - q)

def complete (M Eprev c lag : Nat) : Nat := serviceEnd M Eprev c + lag

/-- Ordinary signed subtraction: the window needs f ≤ c and q ≥ 0.
The lag may even be signed here; graph delays are nonnegative below. -/
theorem signed_completion_window (M Eprev c q lag f : Int)
    (hfc : f ≤ c) (hq : 0 ≤ q) :
    let E := max M Eprev + c
    let C := max (M + f) (E - q)
    let D := E + lag
    C + lag ≤ D ∧ D ≤ C + (q + lag) := by
  dsimp
  omega

/-- No q ≤ E assumption: when E-q is negative, M+f dominates it. -/
theorem commit_cast (M Eprev c q f : Nat) :
    (commit M Eprev c q f : Int) =
      max ((M : Int) + f) (max (M : Int) Eprev + c - q) := by
  unfold commit serviceEnd
  omega

theorem completion_window (M Eprev c q lag f : Nat) (hfc : f ≤ c) :
    commit M Eprev c q f + lag ≤ complete M Eprev c lag ∧
    complete M Eprev c lag ≤ commit M Eprev c q f + (q + lag) := by
  have hs := signed_completion_window (M : Int) Eprev c q lag f
    (by omega) (by omega)
  have hc := commit_cast M Eprev c q f
  dsimp at hs
  unfold complete serviceEnd
  omega

/-- The requested residual bound; C ≤ D follows from the additive lower bound,
so subtraction is not masking an execution with completion before commit. -/
theorem residual_bounds (M Eprev c q lag f : Nat) (hfc : f ≤ c) :
    lag ≤ complete M Eprev c lag - commit M Eprev c q f ∧
    complete M Eprev c lag - commit M Eprev c q f ≤ q + lag := by
  have h := completion_window M Eprev c q lag f hfc
  omega

/-- A parameter envelope yields the edge interval [lagMin, qMax + lagMax]. -/
theorem edge_bounds (M Eprev c q lag f lagMin lagMax qMax : Nat)
    (hfc : f ≤ c) (hlo : lagMin ≤ lag) (hhi : lag ≤ lagMax) (hq : q ≤ qMax) :
    lagMin ≤ qMax + lagMax ∧
    commit M Eprev c q f + lagMin ≤ complete M Eprev c lag ∧
    complete M Eprev c lag ≤ commit M Eprev c q f + (qMax + lagMax) := by
  have h := completion_window M Eprev c q lag f hfc
  omega

/-- Local bridge to Core.Graph: the certified edge supplies its lower constraint
and the completion node's max-over-predecessors upper constraint. Other incoming
lower constraints, source bounds, and a rank certificate remain separate duties. -/
theorem graph_edge (G : Graph V) (τ : Trace V) (u v : V)
    (M Eprev c q lag f lagMin lagMax qMax : Nat)
    (hfc : f ≤ c) (hlo : lagMin ≤ lag) (hhi : lag ≤ lagMax) (hq : q ≤ qMax)
    (hu : τ u = commit M Eprev c q f) (hv : τ v = complete M Eprev c lag)
    (he : (⟨u, lagMin, qMax + lagMax⟩ : Incoming V) ∈ G.preds v) :
    τ u + lagMin ≤ τ v ∧
    τ v ≤ (G.preds v).sup (fun e => τ e.src + e.hi) := by
  obtain ⟨_, hl, hh⟩ := edge_bounds M Eprev c q lag f lagMin lagMax qMax hfc hlo hhi hq
  constructor
  · simpa [hu, hv] using hl
  · have hedge : τ u + (qMax + lagMax) ≤
        (G.preds v).sup (fun e => τ e.src + e.hi) :=
      Finset.le_sup (f := fun e => τ e.src + e.hi) he
    have hbound : τ v ≤ τ u + (qMax + lagMax) := by simpa [hu, hv] using hh
    exact hbound.trans hedge

end GemmProofs.Pipeline.Compute
