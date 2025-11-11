#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrator — hardcoded fries training script → tagged script → per-sentence TTS → stitched MP3.

Public API:
    get_audio_and_timestamps(
        raw_script: str,
        *,
        model: str = "gpt-4.1-mini",
        batch_size: int = 5,
        temperature: float = 0.2,
        max_retries: int = 2,
        openai_api_key: str | None = None,
        add_random_pauses: bool = False,
        pause_p: float = 0.5,
        pause_seed: int | None = None,
        sleep_between_calls: float = 0.0,
    ) -> tuple[list[tuple[bytes, float, float]], bytes]

Returns:
    (timestamps_list, stitched_audio_bytes)
      - timestamps_list: [(segment_bytes, start_s, end_s), ...]
      - stitched_audio_bytes: single MP3 with all segments concatenated

Notes:
  - Requires ELEVEN_API_KEY in the environment for TTS (used by elabs_audio via step2).
  - Requires OPENAI_API_KEY for tagging (step1) unless your step1 is mocked.
"""

from __future__ import annotations
import json
import os
import sys
from typing import List, Tuple

# -----------------------------
# Hardcoded training script
# -----------------------------
RAW_SCRIPT = """
Ok, viens avec moi, je te montre comment on fait les frites ici.
On va le faire ensemble, tu vas voir c’est pas compliqué, mais faut être rigoureux.
Ici, on ne fait pas de frites grasses ou molles, on fait des frites dorées, croustillantes, régulières, à chaque service.

D’abord, regarde les pommes de terre.
On bosse avec de la Bintje, c’est ce qu’il y a de mieux pour les frites, riche en amidon.
On les a épluchées ce matin et coupées à dix millimètres d’épaisseur.
Tu vois, c’est ni trop fin, ni trop gros.
L’idée, c’est qu’elles cuisent vite et restent moelleuses à l’intérieur.

Maintenant, première chose : on les rince à l’eau froide.
Tu mets tout dans le bac, tu remues bien, tu changes l’eau deux fois.
Faut que l’eau soit claire.
C’est pour enlever l’amidon, sinon elles collent entre elles et deviennent molles à la cuisson.
Quand tu les sens propres, tu les égouttes.
Et là, tu fais super gaffe : tu les sèches bien.
Si tu les plonges mouillées, l’eau va éclabousser, tu risques une brûlure et l’huile se détériore plus vite.
Tu prends un torchon propre ou tu passes un coup d’essoreuse, comme ça.
Regarde, là elles sont bien sèches, c’est bon.

On passe à la première cuisson.
Regarde la friteuse, elle est à cent cinquante degrés.
Tu vérifies toujours la température avant de plonger quoi que ce soit.
On ne met jamais un panier plein, la moitié suffit.
Si t’en mets trop, la température chute, et les frites vont boire l’huile.
Tu plonges doucement, pas de gestes brusques.
Tu entends ? Ça doit chanter doucement, pas éclater.
Là, tu les laisses cinq minutes, pas plus.
Elles doivent devenir tendres, souples, légèrement jaunes, mais surtout pas dorées.
C’est ce qu’on appelle la pré-cuisson.

Une fois que c’est bon, tu sors le panier, tu secoues un peu, et tu les mets sur la grille.
Pas de papier tout de suite, faut que la vapeur s’échappe.
Tu les laisses reposer deux, trois minutes.
Tu peux enchaîner le panier suivant pendant ce temps.
Ne sale jamais les frites maintenant, sinon elles ramollissent.

Ensuite, deuxième cuisson, celle qui donne le croustillant.
Tu montes la température à cent quatre-vingts degrés.
Quand l’huile est bien chaude, tu plonges les frites précuites.
Tu vois, elles remontent tout de suite à la surface.
Deux minutes, pas plus.
Tu surveilles la couleur : doré clair, uniforme.
Tu remues un peu le panier, doucement, pour qu’elles cuisent pareil partout.
Quand c’est bon, tu les sors et tu les égouttes.
Et là, tu sales tout de suite, pendant qu’elles sont encore chaudes.
Comme ça le sel accroche bien.
Regarde, elles sont dorées, croustillantes, mais pas grasses.
Tu vois la différence avec les premières ?
Ça, c’est une frite parfaite.

Et surtout, on ne les enferme jamais dans un bac fermé.
Sinon, avec la vapeur, elles ramollissent.
On les garde sur la grille chaude, jamais plus de cinq minutes, sinon on refait un petit passage rapide dans la friteuse.
En service, t’essaies toujours d’avoir un panier d’avance, mais jamais trop, sinon tu jettes.

Tu retiens les trois règles :
huile propre, deux cuissons, sel à la fin.
Si tu fais ça tout le temps, t’auras des frites parfaites à chaque service.

Bon, maintenant, deuxième partie, aussi importante : le nettoyage de la friteuse.
C’est ce que tout le monde bâcle, mais ici, on le fait bien.
Tu attends que l’huile ait un peu refroidi, pas brûlante, sinon tu te fais mal.
Quand c’est tiède, tu prends la passoire à huile et tu filtres.
Tu retires tous les petits morceaux noirs, c’est eux qui font fumer l’huile le lendemain.
Tu verses l’huile propre dans le bidon filtré, et tu gardes ça pour demain si elle est encore claire.
Si elle est foncée ou qu’elle sent fort, tu la jettes, pas de discussion.

Ensuite, tu nettoies la cuve avec le produit prévu, jamais avec de l’eau froide sur une cuve chaude.
Tu grattes les parois, tu rinces bien et tu sèches à fond.
Surtout, pas d’eau qui reste au fond, sinon ça fera exploser l’huile à la prochaine chauffe.
Tu nettoies aussi le pourtour, le couvercle et le sol autour.
Une friteuse propre, c’est un signe de respect et de sécurité.

Tu vois, c’est pas juste de la friture, c’est une méthode.
Quand tu comprends pourquoi on fait chaque étape, tu gagnes du temps, tu réduis les pertes, et tu sers mieux les clients.
Le goût, la texture, la régularité, tout vient de là.
Une cuisine, c’est une chaîne : si la friture est mal faite, tout le reste derrière en pâtit.

Alors retiens bien :
préparation, première cuisson, repos, deuxième cuisson, sel, et nettoyage.
Toujours dans cet ordre, jamais autrement.

Allez, maintenant, à toi.
Je regarde ton premier panier, et je te corrige si besoin.
Tu vas voir, au bout de trois services, tu le feras sans réfléchir.
Et tes frites, elles sortiront toutes pareilles : dorées, croustillantes, parfaites.
""".strip()

RAW_SCRIPT = ' '.join(''.join(ch if ch.isalpha() or ch in ".!?\'" else ' ' for ch in RAW_SCRIPT).split())

# --------------------------
# Step imports
# --------------------------
# Step 1: tagging + optional [pause]
from step1 import (
    add_emotion_tags_to_script,
    randomly_append_pause_tags,
)

# Step 2: per-sentence TTS + stitching
from step2 import synthesize_audio_from_tagged_script

# Optional explicit dependency import to surface errors early
try:
    from elabs_audio import speak_tagged_sentence  # noqa: F401
except Exception:
    # step2 will import what it needs; this is just a soft check
    pass


def get_audio_and_timestamps(
    raw_script: str,
    *,
    model: str = "gpt-4.1-mini",
    batch_size: int = 5,
    temperature: float = 0.2,
    max_retries: int = 2,
    openai_api_key: str | None = None,
    add_random_pauses: bool = False,
    pause_p: float = 0.5,
    pause_seed: int | None = None,
    sleep_between_calls: float = 0.0,
) -> Tuple[List[Tuple[bytes, float, float]], bytes]:
    """
    Full pipeline:
      1) Tag each sentence with an emotion tag (LLM; sentences unchanged).
      2) Optionally append [pause] to ~p of sentences (before punctuation).
      3) ElevenLabs TTS per sentence + stitching to a single MP3.

    Returns:
      timestamps_list, stitched_audio_bytes
    """
    # 1) Tagging
    tagged = add_emotion_tags_to_script(
        raw_text=raw_script,
        model=model,
        batch_size=batch_size,
        temperature=temperature,
        max_retries=max_retries,
        openai_api_key=openai_api_key,
    )

    # 2) Optional random [pause]
    if add_random_pauses:
        tagged = randomly_append_pause_tags(tagged, p=pause_p, seed=pause_seed)

    # 3) TTS per sentence + stitching
    timestamps, stitched = synthesize_audio_from_tagged_script(
        tagged_script=tagged,
        sleep_between_calls=sleep_between_calls,
    )
    return timestamps, stitched


# ----------------
# Demo CLI runner
# ----------------
def main() -> int:
    print("=== Orchestrator demo: hardcoded fries script → audio+timestamps ===")

    # Env checks
    if not os.getenv("ELEVEN_API_KEY"):
        print("Warning: ELEVEN_API_KEY not set — TTS will fail.", file=sys.stderr)
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set — tagging will fail.", file=sys.stderr)

    try:
        timestamps, stitched = get_audio_and_timestamps(
            raw_script=RAW_SCRIPT,
            add_random_pauses=True,      # sprinkle [pause] markers for realism
            pause_p=0.35,
            pause_seed=42,
            sleep_between_calls=0.0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
    except Exception as e:
        print(f"Pipeline failed: {e}", file=sys.stderr)
        return 2

    # Save stitched MP3
    out_mp3 = os.path.abspath("frites_training_stitched.mp3")
    try:
        with open(out_mp3, "wb") as f:
            f.write(stitched)
        print(f"Saved stitched audio → {out_mp3}")
    except Exception as e:
        print(f"Could not write MP3: {e}", file=sys.stderr)
        return 3

    # Save timestamps as JSON for downstream video stitching
    out_json = os.path.abspath("frites_timestamps.json")
    try:
        serializable = [
            # we cannot serialize raw audio segment bytes in JSON, drop them
            {"start": float(s), "end": float(e)} for (_seg, s, e) in timestamps
        ]
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(
                {"segments": serializable, "total_duration": serializable[-1]["end"] if serializable else 0.0},
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Saved timestamps JSON → {out_json}")
        print(f"Segments: {len(timestamps)}")
        if timestamps:
            total = timestamps[-1][2]
            print(f"Total duration: {total:.2f}s")
    except Exception as e:
        print(f"Could not write timestamps JSON: {e}", file=sys.stderr)
        return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
