module
public import GemmProofs.Core.Model

@[expose] public section
namespace GemmProofs.Core

/-- A finite walk carrying the sum of the lower edge offsets. Zero-length
walks allow observation nodes themselves to satisfy a coverage condition. -/
inductive LowerPath (G : Graph V) : V → V → Nat → Prop where
  | refl (v : V) : LowerPath G v v 0
  | step {u v w : V} {d : Nat} (e : Incoming V)
      (hsrc : e.src = v) (he : e ∈ G.preds w) :
      LowerPath G u v d → LowerPath G u w (d + e.lo)

theorem LowerPath.append {G : Graph V} {u v w : V} {a b : Nat}
    (p : LowerPath G u v a) (q : LowerPath G v w b) : LowerPath G u w (a + b) := by
  induction q with
  | refl => simpa using p
  | step e hs he _ ih =>
    simpa [Nat.add_assoc] using LowerPath.step e hs he ih

theorem path_lower {G : Graph V} {τ : Trace V} (hτ : G.ValidLower τ)
    {u v : V} {d : Nat} (path : LowerPath G u v d) : τ u + d ≤ τ v := by
  induction path with
  | refl => omega
  | step e hsrc he _ ih =>
    have edge := hτ.2 _ e he
    rw [hsrc] at edge
    omega

theorem source_path_finish {G : Graph V} {τ : Trace V} (hτ : G.ValidLower τ)
    {s v : V} {d : Nat} (hs : G.preds s = ∅)
    (path : LowerPath G s v d) (O : Finset V) (hv : v ∈ O) :
    G.startLo s + d ≤ finish O τ := by
  have a := hτ.1 s hs
  have b := path_lower hτ path
  have c := le_finish O τ hv
  omega

theorem positive_cycle_impossible {G : Graph V} {v : V} {d : Nat}
    (path : LowerPath G v v d) (hd : 0 < d) : ¬ ∃ τ, G.ValidLower τ := by
  rintro ⟨τ, hτ⟩
  have := path_lower hτ path
  omega

/-- Observation coverage makes the derived completion time bound every event. -/
theorem covered_le_finish {G : Graph V} {τ : Trace V} (hτ : G.ValidLower τ)
    (O : Finset V) (cover : ∀ u, ∃ v ∈ O, ∃ d, LowerPath G u v d)
    (u : V) : τ u ≤ finish O τ := by
  obtain ⟨v, hv, d, path⟩ := cover u
  have := path_lower hτ path
  have := le_finish O τ hv
  omega

end GemmProofs.Core
