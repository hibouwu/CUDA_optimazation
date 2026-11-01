module
public import GemmProofs.Core.Paths
public import GemmProofs.Core.Bounds
public import GemmProofs.Pipeline.Safety

@[expose] public section
namespace GemmProofs.Pipeline
open Core

/-- The four-edge positive cycle exists for every n ≥ 2, not just n = 12. -/
theorem single_slot_impossible (n : Nat) (p : Params) (hn : 2 ≤ n)
    (hd : 0 < p.load.lo) : ¬ ∃ τ, (graph n 1 p true).ValidLower τ := by
  let g0 : Fin n := ⟨0, by omega⟩
  let g1 : Fin n := ⟨1, by omega⟩
  have e1 : incoming g0 .rel zero ∈ (graph n 1 p true).preds (g1, .acq) := by
    simp [graph, g0, g1, incoming]
  have e2 : incoming g1 .acq p.load ∈ (graph n 1 p true).preds (g1, .rdy) := by
    simp [graph]
  have e3 : incoming g1 .rdy zero ∈ (graph n 1 p true).preds (g1, .commit) := by
    simp [graph]
  have e4 : incoming g1 .commit zero ∈ (graph n 1 p true).preds (g0, .rel) := by
    simp [graph, g0, g1, incoming, show 1 < n by omega]
  have cycle := LowerPath.step _ rfl e4
    (LowerPath.step _ rfl e3 (LowerPath.step _ rfl e2
      (LowerPath.step _ rfl e1 (LowerPath.refl (G := graph n 1 p true) (g0, Event.rel)))))
  apply positive_cycle_impossible cycle
  simpa [incoming, zero] using hd

private theorem fixed_offsets (n S issue load mma : Nat) (cross : Bool)
    (v : Node n) (e : Incoming (Node n))
    (he : e ∈ (graph n S (fixed issue load mma) cross).preds v) : e.lo = e.hi := by
  rcases v with ⟨g, k⟩
  cases k <;> simp only [graph] at he
  all_goals try split_ifs at he
  all_goals simp only [Finset.mem_union, Finset.mem_singleton,
    Finset.notMem_empty, false_or, or_false] at he
  all_goals rcases he with rfl | rfl
  all_goals rfl

/-- A valid witness of a fixed-offset DAG is its unique schedule. Therefore it
bounds even traces satisfying only one side of the specification. -/
theorem fixed_witness_bounds (n S issue load mma : Nat) (cross : Bool) (hs : 2 ≤ S)
    (σ : Trace (Node n))
    (hv : (graph n S (fixed issue load mma) cross).ValidLower σ ∧
      (graph n S (fixed issue load mma) cross).ValidUpper σ) :
    (∀ τ, (graph n S (fixed issue load mma) cross).ValidLower τ →
      finish (observations n) σ ≤ finish (observations n) τ) ∧
    (∀ τ, (graph n S (fixed issue load mma) cross).ValidUpper τ →
      finish (observations n) τ ≤ finish (observations n) σ) := by
  let G := graph n S (fixed issue load mma) cross
  have hw := wellFormed n S (fixed issue load mma) cross
  have hr := ranked n S (fixed issue load mma) cross hs
  obtain ⟨ρ, _, unique⟩ := unique_schedule (Node n) G hw
    (by intro v _; rfl) (fixed_offsets n S issue load mma cross) rank hr
  have hlo : σ = schedule G rank hr false :=
    (unique σ hv).trans (unique _ (lower_schedule_valid G hw rank hr)).symm
  have hhi : σ = schedule G rank hr true :=
    (unique σ hv).trans (unique _ (upper_schedule_valid G hw rank hr)).symm
  constructor
  · intro τ hτ
    rw [hlo]
    exact finish_lower G rank hr (observations n) τ hτ
  · intro τ hτ
    rw [hhi]
    exact finish_upper G rank hr (observations n) τ hτ

/-- The acquisition times from ARCH §3.2; only indices below 12 are observed. -/
def acquire4 : Nat → Nat
  | 0 => 0 | 1 => 1 | 2 => 2 | 3 => 3 | 4 => 6 | 5 => 7
  | 6 => 8 | 7 => 10 | 8 => 12 | 9 => 13 | 10 => 14 | _ => 16

def witness4 : Trace (Node 12) := fun (g, k) => match k with
  | .acq => acquire4 g.val
  | .rdy | .commit => acquire4 g.val + 4
  | .cmp => acquire4 g.val + 6
  | .rel => if g.val + 1 < 12 then
      max (acquire4 g.val + 6) (acquire4 (g.val + 1) + 4)
    else acquire4 g.val + 6

def witness4NoCross : Trace (Node 12) := fun (g, k) =>
  let a := 6 * (g.val / 4) + g.val % 4
  match k with
  | .acq => a | .rdy | .commit => a + 4 | .cmp | .rel => a + 6

def witness6 : Trace (Node 12) := fun (g, k) => match k with
  | .acq => g.val | .rdy | .commit => g.val + 4 | .cmp | .rel => g.val + 6

/-- These are kernel-checked finite certificates, not native evaluator assertions. -/
theorem witness4_valid :
    (graph 12 4 (fixed 1 4 2) true).ValidLower witness4 ∧
    (graph 12 4 (fixed 1 4 2) true).ValidUpper witness4 := by
  rw [lower_iff, upper_iff]
  decide

theorem witness4NoCross_valid :
    (graph 12 4 (fixed 1 4 2) false).ValidLower witness4NoCross ∧
    (graph 12 4 (fixed 1 4 2) false).ValidUpper witness4NoCross := by
  rw [lower_iff, upper_iff]
  decide

theorem witness6_valid :
    (graph 12 6 (fixed 1 4 2) true).ValidLower witness6 ∧
    (graph 12 6 (fixed 1 4 2) true).ValidUpper witness6 := by
  rw [lower_iff, upper_iff]
  decide

theorem witness4_finish : finish (observations 12) witness4 = 22 := by decide

theorem witness4NoCross_finish : finish (observations 12) witness4NoCross = 21 := by decide

theorem witness6_finish : finish (observations 12) witness6 = 17 := by decide

theorem four_slot_tight :
    (∃ τ, (graph 12 4 (fixed 1 4 2) true).ValidLower τ ∧
      (graph 12 4 (fixed 1 4 2) true).ValidUpper τ ∧ finish (observations 12) τ = 22) ∧
    (∀ τ, (graph 12 4 (fixed 1 4 2) true).ValidLower τ → 22 ≤ finish (observations 12) τ) ∧
    (∀ τ, (graph 12 4 (fixed 1 4 2) true).ValidUpper τ → finish (observations 12) τ ≤ 22) := by
  have h := fixed_witness_bounds 12 4 1 4 2 true (by decide) witness4 witness4_valid
  rw [witness4_finish] at h
  exact ⟨⟨witness4, witness4_valid.1, witness4_valid.2, witness4_finish⟩, h⟩

theorem four_slot_no_cross_tight :
    (∃ τ, (graph 12 4 (fixed 1 4 2) false).ValidLower τ ∧
      (graph 12 4 (fixed 1 4 2) false).ValidUpper τ ∧ finish (observations 12) τ = 21) ∧
    (∀ τ, (graph 12 4 (fixed 1 4 2) false).ValidLower τ → 21 ≤ finish (observations 12) τ) ∧
    (∀ τ, (graph 12 4 (fixed 1 4 2) false).ValidUpper τ → finish (observations 12) τ ≤ 21) := by
  have h := fixed_witness_bounds 12 4 1 4 2 false (by decide) witness4NoCross witness4NoCross_valid
  rw [witness4NoCross_finish] at h
  exact ⟨⟨witness4NoCross, witness4NoCross_valid.1, witness4NoCross_valid.2,
    witness4NoCross_finish⟩, h⟩

theorem six_slot_tight :
    (∃ τ, (graph 12 6 (fixed 1 4 2) true).ValidLower τ ∧
      (graph 12 6 (fixed 1 4 2) true).ValidUpper τ ∧ finish (observations 12) τ = 17) ∧
    (∀ τ, (graph 12 6 (fixed 1 4 2) true).ValidLower τ → 17 ≤ finish (observations 12) τ) ∧
    (∀ τ, (graph 12 6 (fixed 1 4 2) true).ValidUpper τ → finish (observations 12) τ ≤ 17) := by
  have h := fixed_witness_bounds 12 6 1 4 2 true (by decide) witness6 witness6_valid
  rw [witness6_finish] at h
  exact ⟨⟨witness6, witness6_valid.1, witness6_valid.2, witness6_finish⟩, h⟩

private theorem edge_path {G : Graph V} {u v : V} {d : Nat}
    (he : (⟨u, d, d⟩ : Incoming V) ∈ G.preds v) : LowerPath G u v d := by
  simpa using LowerPath.step ⟨u, d, d⟩ rfl he (LowerPath.refl u)

private theorem stage_path (S : Nat) (cross : Bool) (g : Fin 12) :
    LowerPath (graph 12 S (fixed 1 4 2) cross) (g, .acq) (g, .rel) 6 := by
  have a : LowerPath (graph 12 S (fixed 1 4 2) cross) (g, .acq) (g, .rdy) 4 :=
    edge_path (by simp [graph, incoming, fixed])
  have b : LowerPath (graph 12 S (fixed 1 4 2) cross) (g, .rdy) (g, .commit) 0 :=
    edge_path (by simp [graph, incoming, zero])
  have c : LowerPath (graph 12 S (fixed 1 4 2) cross) (g, .commit) (g, .cmp) 2 :=
    edge_path (by simp [graph, incoming, fixed])
  have d : LowerPath (graph 12 S (fixed 1 4 2) cross) (g, .cmp) (g, .rel) 0 :=
    edge_path (by simp [graph, incoming, zero])
  exact ((a.append b).append c).append d

/-- ARCH's cross-slot path, explicitly checked edge by edge. -/
theorem critical_path22 :
    LowerPath (graph 12 4 (fixed 1 4 2) true) (0, .acq) (11, .rel) 22 := by
  have a := stage_path 4 true 0
  have b : LowerPath (graph 12 4 (fixed 1 4 2) true) (0, .rel) (4, .acq) 0 :=
    edge_path (by decide)
  have c : LowerPath (graph 12 4 (fixed 1 4 2) true) (4, .acq) (4, .rdy) 4 :=
    edge_path (by decide)
  have d : LowerPath (graph 12 4 (fixed 1 4 2) true) (4, .rdy) (4, .commit) 0 :=
    edge_path (by decide)
  have e : LowerPath (graph 12 4 (fixed 1 4 2) true) (4, .commit) (3, .rel) 0 :=
    edge_path (by decide)
  have f : LowerPath (graph 12 4 (fixed 1 4 2) true) (3, .rel) (7, .acq) 0 :=
    edge_path (by decide)
  have g := stage_path 4 true 7
  have h : LowerPath (graph 12 4 (fixed 1 4 2) true) (7, .rel) (11, .acq) 0 :=
    edge_path (by decide)
  have i := stage_path 4 true 11
  exact (((((((a.append b).append c).append d).append e).append f).append g).append h).append i

/-- Successive acquisition edges give the launch-prefix path without enumeration. -/
private theorem issue_path (S : Nat) (cross : Bool) (g : Fin 12) :
    LowerPath (graph 12 S (fixed 1 4 2) cross) (0, .acq) (g, .acq) g.val := by
  induction g using (measure (fun g : Fin 12 => g.val)).wf.induction with
  | h g ih =>
    by_cases h0 : g.val = 0
    · have hg : g = 0 := Fin.ext h0
      subst g
      exact LowerPath.refl _
    · let prev : Fin 12 := ⟨g.val - 1, by omega⟩
      have hp : prev.val < g.val := by dsimp [prev]; omega
      have a := ih prev hp
      have b : LowerPath (graph 12 S (fixed 1 4 2) cross) (prev, .acq) (g, .acq) 1 :=
        edge_path (by
          simp only [graph]
          apply Finset.mem_union_left
          simp only [dite_eq_left (show 0 < g.val by omega)]
          exact Finset.mem_singleton.mpr rfl)
      have sum : prev.val + 1 = g.val := by dsimp [prev]; omega
      simpa only [sum] using a.append b

theorem critical_path17 :
    LowerPath (graph 12 6 (fixed 1 4 2) true) (0, .acq) (11, .rel) 17 :=
  (issue_path 6 true 11).append (stage_path 6 true 11)

theorem critical_path21 :
    LowerPath (graph 12 4 (fixed 1 4 2) false) (0, .acq) (11, .rel) 21 := by
  have a := issue_path 4 false 3
  have b := stage_path 4 false 3
  have c : LowerPath (graph 12 4 (fixed 1 4 2) false) (3, .rel) (7, .acq) 0 :=
    edge_path (by decide)
  have d := stage_path 4 false 7
  have e : LowerPath (graph 12 4 (fixed 1 4 2) false) (7, .rel) (11, .acq) 0 :=
    edge_path (by decide)
  have f := stage_path 4 false 11
  exact ((((a.append b).append c).append d).append e).append f

theorem slot_comparison (τ4 τ6 : Trace (Node 12))
    (h4 : (graph 12 4 (fixed 1 4 2) true).ValidLower τ4)
    (h6 : (graph 12 6 (fixed 1 4 2) true).ValidUpper τ6) :
    finish (observations 12) τ6 < finish (observations 12) τ4 := by
  have := four_slot_tight.2.1 τ4 h4
  have := six_slot_tight.2.2 τ6 h6
  omega

theorem cross_edge_comparison (τCross τNoCross : Trace (Node 12))
    (hc : (graph 12 4 (fixed 1 4 2) true).ValidLower τCross)
    (hn : (graph 12 4 (fixed 1 4 2) false).ValidUpper τNoCross) :
    finish (observations 12) τNoCross < finish (observations 12) τCross := by
  have := four_slot_tight.2.1 τCross hc
  have := four_slot_no_cross_tight.2.2 τNoCross hn
  omega

/-- The same six-slot witness also satisfies the graph with the cross edge removed. -/
theorem witness6NoCross_valid :
    (graph 12 6 (fixed 1 4 2) false).ValidLower witness6 ∧
    (graph 12 6 (fixed 1 4 2) false).ValidUpper witness6 := by
  rw [lower_iff, upper_iff]
  decide

theorem six_slot_no_cross_tight :
    (∃ τ, (graph 12 6 (fixed 1 4 2) false).ValidLower τ ∧
      (graph 12 6 (fixed 1 4 2) false).ValidUpper τ ∧ finish (observations 12) τ = 17) ∧
    (∀ τ, (graph 12 6 (fixed 1 4 2) false).ValidLower τ → 17 ≤ finish (observations 12) τ) ∧
    (∀ τ, (graph 12 6 (fixed 1 4 2) false).ValidUpper τ → finish (observations 12) τ ≤ 17) := by
  have h := fixed_witness_bounds 12 6 1 4 2 false (by decide) witness6 witness6NoCross_valid
  rw [witness6_finish] at h
  exact ⟨⟨witness6, witness6NoCross_valid.1, witness6NoCross_valid.2, witness6_finish⟩, h⟩

end GemmProofs.Pipeline
