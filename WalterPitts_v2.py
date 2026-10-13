#!/usr/bin/env python3
"""Walter Pitts V2: empirijski zajednicki prelazi i pragovna logika."""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


DEFAULTS = (
    "/Users/4c/Desktop/GHQ/data/loto7_4700_k81.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4700_k81_loto_2972.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4700_k81_loto_plus_1728.csv",
)

WINDOW = 384
STARTS = 12

# Sirina slicnosti, geometrijski doprinos, doprinos celog rasporeda.
PARAMS = (
    (.5, .1, .1),
    (1., .1, .1),
    (1., .25, .1),
    (2., .25, .1),
)


def load(path):
    rows = []

    with Path(path).open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if not row or all(not v.strip() for v in row):
                continue

            try:
                values = [int(v.strip()) for v in row]
            except ValueError as error:
                raise ValueError(
                    f"Red {line}: ocekujem cele brojeve."
                ) from error

            if (
                len(values) != 7
                or values != sorted(values)
                or len(set(values)) != 7
                or not all(1 <= v <= 39 for v in values)
            ):
                raise ValueError(
                    f"Red {line}: neispravna kombinacija 7/39."
                )

            rows.append(values)

    return np.asarray(rows, dtype=int).reshape(-1, 7)


def gaps(rows):
    rows = np.atleast_2d(rows)

    edges = np.column_stack((
        np.zeros(len(rows)),
        rows,
        np.full(len(rows), 40),
    ))

    return np.diff(edges, axis=1) - 1.


def replacements(row):
    unused = np.setdiff1d(np.arange(1, 40), row)
    result = np.tile(row, (224, 1))

    for j in range(7):
        result[j * 32:(j + 1) * 32, j] = unused

    return np.sort(result, axis=1)


def hits(prediction, actual):
    return len(set(prediction).intersection(actual))


def neuron(excitation, threshold, inhibition=None):
    active = np.sum(excitation, axis=-1) >= threshold

    if inhibition is not None:
        active &= ~np.any(inhibition, axis=-1)

    return active


def geometry(g):
    g = np.atleast_2d(g)

    current = np.ones(len(g), dtype=int)
    longest = current.copy()

    for j in range(1, 7):
        current = np.where(
            g[:, j] == 0, current + 1, 1
        )
        longest = np.maximum(longest, current)

    return np.column_stack((
        longest,
        np.max(g[:, 1:7], axis=1),
    ))


def layout(g):
    return np.column_stack((g, geometry(g)))


def logic(g, reference, previous, earlier, cutoff):
    g = np.atleast_2d(g)

    delta = g - reference
    up = delta > 0
    down = delta < 0
    large = np.abs(delta) >= cutoff

    last_up = np.broadcast_to(previous > 0, up.shape)
    last_down = np.broadcast_to(previous < 0, up.shape)
    old_up = np.broadcast_to(earlier > 0, up.shape)
    old_down = np.broadcast_to(earlier < 0, up.shape)

    def AND(a, b):
        return neuron(
            np.stack((a, b), axis=-1), 2
        )

    return np.column_stack((
        up,
        down,
        large,

        AND(up, last_up),
        AND(down, last_down),

        AND(up, last_down),
        AND(down, last_up),

        AND(up[:, :-1], down[:, 1:]),
        AND(down[:, :-1], up[:, 1:]),

        neuron(
            np.stack((up, last_up), axis=-1),
            2,
            old_down[..., None],
        ),
        neuron(
            np.stack((down, last_down), axis=-1),
            2,
            old_up[..., None],
        ),

        neuron(
            np.stack((up, last_down, old_down), axis=-1),
            3,
        ),
        neuron(
            np.stack((down, last_up, old_up), axis=-1),
            3,
        ),

        neuron(
            np.stack((up, last_up, old_up), axis=-1),
            2,
        ),
        neuron(
            np.stack((down, last_down, old_down), axis=-1),
            2,
        ),
    )).astype(float)


def l1(a, b, scale):
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)

    distance = np.zeros((len(a), len(b)))

    for j in range(a.shape[1]):
        distance += (
            np.abs(a[:, j, None] - b[None, :, j])
            / scale[j]
        )

    return distance


def bandwidth(values, scale):
    normalized = values / scale

    nearest = cKDTree(normalized).query(
        normalized, k=2, p=1
    )[0][:, 1]

    positive = nearest[nearest > 1e-12]

    value = (
        float(np.median(positive))
        if len(positive)
        else 1.
    )

    return max(value, 1.)


def prepare(history):
    needed = WINDOW + 3

    if len(history) < needed:
        raise ValueError(
            f"Potrebno je najmanje {needed} prethodnih redova."
        )

    all_g = gaps(history)
    g = all_g[-needed:]

    target = g[3:]
    reference = g[2:-1]

    previous = g[2:-1] - g[1:-2]
    earlier = g[1:-2] - g[:-3]
    delta = target - reference

    # Skale osobina; ne formiraju ciljnu kombinaciju.
    delta_scale = np.maximum(
        np.median(np.abs(delta), axis=0), 1.
    )

    target_layout = layout(target)

    layout_scale = np.maximum(
        np.median(
            np.abs(np.diff(layout(g), axis=0)),
            axis=0,
        ),
        1.,
    )

    actual_logic = logic(
        target,
        reference,
        previous,
        earlier,
        delta_scale,
    )

    current_previous = g[-1] - g[-2]
    current_earlier = g[-2] - g[-3]

    context_distance = l1(
        layout(g[-1:]),
        layout(reference),
        layout_scale,
    )[0]

    # Redosled prethodna dva prelaza ostaje u kontekstu.
    context_distance += (
        np.sum(
            (previous > 0) != (current_previous > 0),
            axis=1,
        )
        + np.sum(
            (previous < 0) != (current_previous < 0),
            axis=1,
        )
        + np.sum(
            (earlier > 0) != (current_earlier > 0),
            axis=1,
        )
        + np.sum(
            (earlier < 0) != (current_earlier < 0),
            axis=1,
        )
    ) / 8.

    positive = context_distance[
        context_distance > 1e-12
    ]

    context_width = max(
        float(np.quantile(positive, .25))
        if len(positive)
        else 1.,
        1.,
    )

    differences = np.sum(
        np.abs(np.diff(actual_logic, axis=0)),
        axis=1,
    )

    positive_logic = differences[differences > 0]

    logic_width = max(
        float(np.median(positive_logic))
        if len(positive_logic)
        else 1.,
        1.,
    )

    # Zajednicka raspodela oblika iz cele dostupne istorije.
    # Ne racunaju se frekvencije pojedinacnih loto brojeva.
    shapes, counts = np.unique(
        geometry(all_g),
        axis=0,
        return_counts=True,
    )

    shape_mass = counts / counts.sum()

    shape_scale = layout_scale[-2:].copy()
    shape_scale[0] = 1.

    # Tacni istorijski prelazi preneti na sadasnje stanje.
    # Nema proseka prelaza ili prosecne ciljne sedmorke.
    translated = g[-1] + delta
    valid = np.all(translated >= 0, axis=1)

    translated = np.cumsum(
        translated[valid, :7] + 1,
        axis=1,
    ).astype(int)

    seeds = np.unique(
        np.vstack((
            history[-WINDOW:],
            translated,
        )),
        axis=0,
    )

    return dict(
        reference=g[-1],
        previous=current_previous,
        earlier=current_earlier,
        delta=delta,
        delta_scale=delta_scale,
        actual_logic=actual_logic,
        layout=target_layout,
        layout_scale=layout_scale,
        delta_width=bandwidth(delta, delta_scale),
        layout_width=bandwidth(
            target_layout, layout_scale
        ),
        context=context_distance,
        context_width=context_width,
        logic_width=logic_width,
        shapes=shapes,
        shape_mass=shape_mass,
        shape_scale=shape_scale,
        seeds=seeds,
    )


def fit(p, param):
    width, beta, gamma = param

    weight = (
        1. + p["context"]
        / (p["context_width"] * width)
    ) ** -4

    weight /= weight.sum()

    return p, width, beta, gamma, weight


def score_g(g, model):
    p, width, beta, gamma, weight = model

    candidate_delta = g - p["reference"]

    candidate_logic = logic(
        g,
        p["reference"],
        p["previous"],
        p["earlier"],
        p["delta_scale"],
    )

    # Udaljenost izmedju celih logickih obrazaca.
    logic_distance = (
        candidate_logic.sum(axis=1)[:, None]
        + p["actual_logic"].sum(axis=1)[None, :]
        - 2. * candidate_logic @ p["actual_logic"].T
    )

    logic_distance = np.maximum(
        logic_distance, 0.
    )

    transition_distance = l1(
        candidate_delta,
        p["delta"],
        p["delta_scale"],
    )

    transition_distance /= (
        p["delta_width"] * width
    )

    transition_distance += (
        logic_distance
        / (p["logic_width"] * width)
    )

    conditional = (
        (1. + transition_distance) ** -4
    ) @ weight

    shape_distance = l1(
        geometry(g),
        p["shapes"],
        p["shape_scale"],
    )

    geometric = (
        (1. + shape_distance) ** -4
    ) @ p["shape_mass"]

    state_distance = l1(
        layout(g),
        p["layout"],
        p["layout_scale"],
    )

    state_density = np.mean(
        (
            1. + state_distance
            / (p["layout_width"] * width)
        ) ** -4,
        axis=1,
    )

    # Ogranicene empirijske slicnosti.
    # Nema polinoma koji nagradjuju ekstremne rasporede.
    tiny = np.finfo(float).tiny

    return (
        np.log(np.maximum(conditional, tiny))
        + beta * np.log(np.maximum(geometric, tiny))
        + gamma * np.log(np.maximum(state_density, tiny))
    )


def scores(rows, model):
    rows = np.atleast_2d(rows)
    result = np.empty(len(rows))

    for start in range(0, len(rows), 256):
        stop = min(start + 256, len(rows))

        result[start:stop] = score_g(
            gaps(rows[start:stop]),
            model,
        )

    if not np.all(np.isfinite(result)):
        raise ArithmeticError(
            "Neispravan skor empirijske mreze."
        )

    return result


def search(model):
    p = model[0]
    seeds = p["seeds"]

    order = np.argsort(
        -scores(seeds, model),
        kind="stable",
    )[:STARTS]

    current = seeds[order].copy()
    steps = 0

    while True:
        moved = False

        for j, row in enumerate(current):
            candidates = replacements(row)
            values = scores(candidates, model)

            best = int(np.argmax(values))

            if (
                values[best]
                > scores(row, model)[0] + 1e-10
            ):
                current[j] = candidates[best]
                moved = True
                steps += 1

        if not moved:
            break

    values = scores(current, model)
    best = int(np.argmax(values))

    return (
        tuple(map(int, current[best])),
        float(values[best]),
        steps,
    )


def predict(p, param):
    return search(fit(p, param))


def run_file(path, validation=48, test=64):
    history = load(path)
    split = len(history) - test

    if (
        validation < 16
        or test < 16
        or split - validation < WINDOW + 3
    ):
        raise ValueError(
            "Nedovoljno podataka ili prekratka provera."
        )

    totals = np.zeros(
        len(PARAMS), dtype=int
    )

    # Izbor parametara iskljucivo pre testnog perioda.
    for t in range(
        split - validation, split
    ):
        p = prepare(history[:t])

        for j, param in enumerate(PARAMS):
            totals[j] += hits(
                predict(p, param)[0],
                history[t],
            )

    param = PARAMS[int(np.argmax(totals))]

    matches = []
    previous = []
    predicted_shapes = []
    predicted_rows = []
    replay_count = 0

    # Zamrznut parametar; svaki korak koristi samo proslost.
    for t in range(split, len(history)):
        prediction, _, _ = predict(
            prepare(history[:t]),
            param,
        )

        matches.append(
            hits(prediction, history[t])
        )

        previous.append(
            hits(history[t - 1], history[t])
        )

        predicted_shapes.append(
            geometry(gaps(prediction))[0]
        )

        predicted_rows.append(prediction)

        replay_count += bool(
            np.any(
                np.all(
                    history[:t] == prediction,
                    axis=1,
                )
            )
        )

    predicted_shapes = np.asarray(
        predicted_shapes
    )

    actual_shapes = geometry(
        gaps(history[split:])
    )

    print("\nCSV:", path)
    print(
        "CSV mora biti hronoloski: najstarije kolo prvo."
    )

    print(
        f"Validacija={validation}; "
        f"odvojeni test={test}; "
        f"parametri={param}"
    )

    print(
        f"WALTER V2: prosek={np.mean(matches):.4f}; "
        f"pogoci 0..7="
        f"{np.bincount(matches, minlength=8).tolist()}"
    )

    print(
        "Kontrola: prethodno izvuceno kolo:",
        f"{np.mean(previous):.4f}",
    )

    print(
        "Nizovi od najmanje 5: model/stvarna kola =",
        int(np.sum(predicted_shapes[:, 0] >= 5)),
        int(np.sum(actual_shapes[:, 0] >= 5)),
    )

    print(
        "Unutrasnja praznina najmanje 20: model/stvarna kola =",
        int(np.sum(predicted_shapes[:, 1] >= 20)),
        int(np.sum(actual_shapes[:, 1] >= 20)),
    )

    print(
        "Istorijska ponavljanja na testu:",
        replay_count,
    )

    unique_rows, counts = np.unique(
        np.asarray(predicted_rows),
        axis=0,
        return_counts=True,
    )

    print(
        "Razlicitih testnih izbora:",
        len(unique_rows),
        "; najvece ponavljanje iste sedmorke:",
        int(counts.max()),
    )

    prediction, score, steps = predict(
        prepare(history), param
    )

    shape = geometry(
        gaps(prediction)
    )[0].astype(int)

    replay = bool(
        np.any(
            np.all(
                history == prediction,
                axis=1,
            )
        )
    )

    print(
        f"SLEDECI RED {len(history) + 1}:",
        ",".join(map(str, prediction)),
    )

    print(
        f"Skor={score:.10f}; "
        f"svi koraci={steps}; "
        f"istorijsko ponavljanje={replay}"
    )

    print(
        f"Najduzi uzastopni niz={shape[0]}; "
        f"najveca unutrasnja praznina={shape[1]}"
    )

    print(
        "Skor je empirijska podrska, "
        "nije kalibrisana verovatnoca pogotka."
    )

    print(
        "Pretraga: istorijska i preneta polazista, "
        "sve zamene jednog broja."
    )

    print(
        "Globalni maksimum nije potvrden; "
        "nema zabrane grupisanih brojeva."
    )

    return prediction


def self_test():
    truth = np.array((
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ), dtype=bool)

    assert np.array_equal(
        neuron(truth, 2),
        (False, False, False, True),
    )

    assert np.array_equal(
        neuron(truth, 1),
        (False, True, True, True),
    )

    assert not neuron(
        np.ones((1, 20), dtype=bool),
        1,
        np.ones((1, 1), dtype=bool),
    )[0]

    base = np.array((
        (1, 2, 3, 4, 5, 6, 7),
        (4, 12, 18, 28, 35, 36, 37),
        (33, 34, 35, 36, 37, 38, 39),
        (4, 8, 13, 18, 23, 28, 34),
    ))

    g = gaps(base)

    assert np.all(g.sum(axis=1) == 32)

    assert np.array_equal(
        np.cumsum(g[:, :7] + 1, axis=1),
        base,
    )

    assert np.array_equal(
        geometry(g)[:, 0],
        (7, 3, 7, 1),
    )

    h = np.tile(base, (100, 1))
    p = prepare(h)
    model = fit(p, PARAMS[1])

    assert np.isclose(model[4].sum(), 1.)
    assert np.isclose(p["shape_mass"].sum(), 1.)
    assert np.all(scores(base, model) <= 1e-12)

    batch = logic(
        g,
        g[-1],
        g[-1] - g[-2],
        g[-2] - g[-3],
        p["delta_scale"],
    )

    for j in range(len(g)):
        individual = logic(
            g[j:j + 1],
            g[-1],
            g[-1] - g[-2],
            g[-2] - g[-3],
            p["delta_scale"],
        )[0]

        assert np.array_equal(
            batch[j], individual
        )

    prediction, score, _ = predict(
        p, PARAMS[1]
    )

    assert prediction == predict(
        prepare(h.copy()),
        PARAMS[1],
    )[0]

    assert (
        len(set(prediction)) == 7
        and 1 <= min(prediction)
        <= max(prediction) <= 39
    )

    assert (
        scores(
            replacements(np.array(prediction)),
            model,
        ).max()
        <= score + 1e-9
    )

    future = h.copy()
    future[-1] = base[1]

    assert predict(
        prepare(h[:-1]), PARAMS[1]
    )[0] == predict(
        prepare(future[:-1]), PARAMS[1]
    )[0]

    print(
        "Self-test OK: logika, inhibicija, "
        "geometrija, skor i hronologija."
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "csv",
        nargs="*",
        help="Podrazumevano sva tri nova CSV fajla.",
    )

    parser.add_argument(
        "--validation",
        type=int,
        default=48,
    )

    parser.add_argument(
        "--test",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()

    seen = {}

    for path in args.csv or DEFAULTS:
        prediction = run_file(
            path,
            args.validation,
            args.test,
        )

        if prediction in seen:
            print(
                "ISTI IZLAZ ZA DVA FAJLA:",
                seen[prediction],
                "i",
                path,
            )

        seen[prediction] = path


if __name__ == "__main__":
    main()



"""
CSV: /Users/4c/Desktop/GHQ/data/loto7_4700_k81.csv
CSV mora biti hronoloski: najstarije kolo prvo.
Validacija=48; odvojeni test=64; parametri=(0.5, 0.1, 0.1)
WALTER V2: prosek=1.2500; pogoci 0..7=[13, 28, 17, 6, 0, 0, 0, 0]
Kontrola: prethodno izvuceno kolo: 1.2344
Nizovi od najmanje 5: model/stvarna kola = 0 0
Unutrasnja praznina najmanje 20: model/stvarna kola = 0 1
Istorijska ponavljanja na testu: 0
Razlicitih testnih izbora: 64 ; najvece ponavljanje iste sedmorke: 1
SLEDECI RED 4701: 2,6,18,24,27,34,36
Skor=-6.4887691301; svi koraci=7; istorijsko ponavljanje=False
Najduzi uzastopni niz=1; najveca unutrasnja praznina=11
Skor je empirijska podrska, nije kalibrisana verovatnoca pogotka.
Pretraga: istorijska i preneta polazista, sve zamene jednog broja.
Globalni maksimum nije potvrden; nema zabrane grupisanih brojeva.



CSV: /Users/4c/Desktop/GHQ/data/loto7_4700_k81_loto_2972.csv
CSV mora biti hronoloski: najstarije kolo prvo.
Validacija=48; odvojeni test=64; parametri=(1.0, 0.25, 0.1)
WALTER V2: prosek=1.4844; pogoci 0..7=[9, 25, 21, 8, 1, 0, 0, 0]
Kontrola: prethodno izvuceno kolo: 1.3281
Nizovi od najmanje 5: model/stvarna kola = 0 0
Unutrasnja praznina najmanje 20: model/stvarna kola = 0 1
Istorijska ponavljanja na testu: 9
Razlicitih testnih izbora: 64 ; najvece ponavljanje iste sedmorke: 1
SLEDECI RED 2973: 7,8,18,23,33,36,37
Skor=-5.9077992876; svi koraci=36; istorijsko ponavljanje=False
Najduzi uzastopni niz=2; najveca unutrasnja praznina=9
Skor je empirijska podrska, nije kalibrisana verovatnoca pogotka.
Pretraga: istorijska i preneta polazista, sve zamene jednog broja.
Globalni maksimum nije potvrden; nema zabrane grupisanih brojeva.



CSV: /Users/4c/Desktop/GHQ/data/loto7_4700_k81_loto_plus_1728.csv
CSV mora biti hronoloski: najstarije kolo prvo.
Validacija=48; odvojeni test=64; parametri=(0.5, 0.1, 0.1)
WALTER V2: prosek=1.3750; pogoci 0..7=[8, 29, 22, 5, 0, 0, 0, 0]
Kontrola: prethodno izvuceno kolo: 1.2969
Nizovi od najmanje 5: model/stvarna kola = 0 0
Unutrasnja praznina najmanje 20: model/stvarna kola = 0 1
Istorijska ponavljanja na testu: 0
Razlicitih testnih izbora: 64 ; najvece ponavljanje iste sedmorke: 1
SLEDECI RED 1729: 15,18,23,31,35,38,39
Skor=-7.5419823544; svi koraci=18; istorijsko ponavljanje=False
Najduzi uzastopni niz=2; najveca unutrasnja praznina=7
Skor je empirijska podrska, nije kalibrisana verovatnoca pogotka.
Pretraga: istorijska i preneta polazista, sve zamene jednog broja.
Globalni maksimum nije potvrden; nema zabrane grupisanih brojeva.
"""



"""
Uklonjen je polinomski skor koji je nagrađivao ekstremne rasporede. 
Model koristi empirijsku podršku celim prelazima i prijavljuje grupisanje, praznine i ponavljanja.
"""
