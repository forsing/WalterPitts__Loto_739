#!/usr/bin/env python3
"""McCulloch-Pitts V1: pragovna logicka mreza za skor cele kombinacije."""

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

WINDOW, STARTS = 96, 12
ALPHAS = (.01, .1, 1.)
DIM = 166


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
        np.full(len(rows), 40)
    ))

    return np.diff(edges, axis=1) - 1.


def replacements(row):
    unused = np.setdiff1d(np.arange(1, 40), row)
    result = np.tile(row, (224, 1))

    for j in range(7):
        result[j * 32:(j + 1) * 32, j] = unused

    return np.sort(result, axis=1)


def support(history):
    tree = cKDTree(gaps(history) / 5.)

    distance = tree.query(
        gaps(history) / 5., k=2, p=1
    )[0][:, 1]

    return tree, max(
        float(np.quantile(distance, .75)),
        1e-12
    )


def allowed(rows, tree, radius):
    distance = tree.query(
        gaps(rows) / 5., p=1
    )[0]

    return distance <= radius + 1e-10


def neuron(excitation, threshold, inhibition=None):
    active = np.sum(
        excitation, axis=-1
    ) >= threshold

    if inhibition is not None:
        active = active & ~np.any(
            inhibition, axis=-1
        )

    return active


def gap_bits(g):
    # Potpuno, bezgubitno kodiranje razmaka 0..32.
    # Interval: pobuda donjom granicom i veto gornjom.
    # Binarni zapis je reprezentacija, ne raspodela.
    result = []

    for bit in range(6):
        width = 1 << bit
        starts = np.arange(
            width, 33, 2 * width
        )

        lower = g[:, :, None] >= starts
        upper = g[:, :, None] >= starts + width

        interval = neuron(
            lower[..., None],
            1,
            upper[..., None]
        )

        result.append(
            neuron(interval, 1)
        )

    return np.stack(
        result, axis=-1
    ).reshape(len(g), 48)


def context(g, temporal=True):
    changes = np.diff(g, axis=0)

    # Prag vece promene dobija se samo iz proslosti.
    cutoff = np.maximum(
        np.ceil(
            np.quantile(
                np.abs(changes[-WINDOW:]),
                .75,
                axis=0
            )
        ),
        1.
    )

    return (
        g[-1],
        changes[-1],
        changes[-2],
        cutoff,
        temporal
    )


def features(g, ctx):
    g = np.atleast_2d(g)

    (
        reference,
        previous,
        earlier,
        cutoff,
        temporal
    ) = ctx

    delta = g - reference

    up = delta > 0
    down = delta < 0
    large = np.abs(delta) >= cutoff

    last_up = np.broadcast_to(
        previous > 0, up.shape
    )

    last_down = np.broadcast_to(
        previous < 0, up.shape
    )

    old_up = np.broadcast_to(
        earlier > 0, up.shape
    )

    old_down = np.broadcast_to(
        earlier < 0, up.shape
    )

    def AND(a, b):
        return neuron(
            np.stack((a, b), axis=-1),
            2
        )

    continuation = np.column_stack((
        AND(up, last_up),
        AND(down, last_down)
    ))

    reversal = np.column_stack((
        AND(up, last_down),
        AND(down, last_up)
    ))

    adjacent = np.column_stack((
        AND(up[:, :-1], down[:, 1:]),
        AND(down[:, :-1], up[:, 1:])
    ))

    inhibited = np.column_stack((
        neuron(
            np.stack((up, last_up), axis=-1),
            2,
            old_down[..., None]
        ),
        neuron(
            np.stack((down, last_down), axis=-1),
            2,
            old_up[..., None]
        )
    ))

    sustained = np.column_stack((
        neuron(
            np.stack(
                (up, last_down, old_down),
                axis=-1
            ),
            3
        ),
        neuron(
            np.stack(
                (down, last_up, old_up),
                axis=-1
            ),
            3
        )
    ))

    majority = np.column_stack((
        neuron(
            np.stack(
                (up, last_up, old_up),
                axis=-1
            ),
            2
        ),
        neuron(
            np.stack(
                (down, last_down, old_down),
                axis=-1
            ),
            2
        )
    ))

    if not temporal:
        continuation = np.zeros_like(continuation)
        reversal = np.zeros_like(reversal)
        inhibited = np.zeros_like(inhibited)
        sustained = np.zeros_like(sustained)
        majority = np.zeros_like(majority)

    return np.column_stack((
        gap_bits(g),
        up,
        down,
        large,
        continuation,
        reversal,
        adjacent,
        inhibited,
        sustained,
        majority
    )).astype(float)


def prepare(history, temporal=True):
    needed = WINDOW + 3

    if len(history) < needed:
        raise ValueError(
            f"Potrebno je najmanje "
            f"{needed} prethodnih redova."
        )

    h = history[-needed:]
    g = gaps(h)

    gram = np.zeros((DIM, DIM))
    rhs = np.zeros(DIM)

    for t in range(3, len(h)):
        ctx = context(
            g[:t], temporal
        )

        actual = features(
            g[t:t + 1], ctx
        )

        other = features(
            gaps(replacements(h[t])),
            ctx
        )

        contrast = actual - other

        gram += (
            contrast.T @ contrast
            / len(contrast)
        )

        rhs += contrast.mean(axis=0)

    gram /= WINDOW
    rhs /= WINDOW

    # Numericka skala logickih osobina.
    # Ne prosecuje ciljne brojeve ili razmake.
    scale = np.maximum(
        np.sqrt(np.diag(gram)),
        1e-8
    )

    return (
        gram / scale[:, None] / scale[None, :],
        rhs / scale,
        scale,
        context(g, temporal),
        h[-WINDOW:]
    )


def fit(p, alpha):
    coefficient = (
        np.linalg.solve(
            p[0] + alpha * np.eye(DIM),
            p[1]
        ) / p[2]
    )

    return coefficient, p[3]


def scores(rows, model):
    coefficient, ctx = model

    value = (
        features(gaps(rows), ctx)
        @ coefficient
    )

    if not np.all(np.isfinite(value)):
        raise ArithmeticError(
            "Neispravan skor logicke mreze."
        )

    return value


def search(history, model):
    tree, radius = support(history)

    seeds = np.unique(
        history, axis=0
    )

    order = np.argsort(
        -scores(seeds, model),
        kind="stable"
    )[:STARTS]

    current = seeds[order].copy()
    iterations = 0

    while True:
        moved = False

        for j, row in enumerate(current):
            candidates = replacements(row)

            candidates = candidates[
                allowed(
                    candidates,
                    tree,
                    radius
                )
            ]

            if not len(candidates):
                continue

            values = scores(
                candidates, model
            )

            best = int(np.argmax(values))

            if (
                values[best]
                > scores(row, model)[0] + 1e-10
            ):
                current[j] = candidates[best]
                moved = True

        if not moved:
            break

        iterations += 1

    values = scores(current, model)
    best = int(np.argmax(values))

    return (
        tuple(map(int, current[best])),
        float(values[best]),
        iterations
    )


def predict(p, alpha):
    return search(
        p[4], fit(p, alpha)
    )


def hits(prediction, actual):
    return len(
        set(prediction).intersection(actual)
    )


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

    labels = {
        True: "MCCULLOCH-PITTS: LOGIKA I VREMENSKI USLOVI",
        False: "Kontrola: bez vremenskih logickih uslova"
    }

    chosen = {}

    # Izbor parametara iskljucivo pre testnog perioda.
    for mode in (True, False):
        totals = np.zeros(
            len(ALPHAS), dtype=int
        )

        for t in range(
            split - validation, split
        ):
            p = prepare(
                history[:t], mode
            )

            for j, alpha in enumerate(ALPHAS):
                totals[j] += hits(
                    predict(p, alpha)[0],
                    history[t]
                )

        chosen[mode] = ALPHAS[
            int(np.argmax(totals))
        ]

    result = {
        True: [],
        False: [],
        "previous": []
    }

    # Parametri su zamrznuti tokom odvojenog testa.
    # Svako predvidjanje koristi samo prethodna kola.
    for t in range(split, len(history)):
        for mode in (True, False):
            p = prepare(
                history[:t], mode
            )

            prediction = predict(
                p, chosen[mode]
            )[0]

            result[mode].append(
                hits(prediction, history[t])
            )

        result["previous"].append(
            hits(history[t - 1], history[t])
        )

    print("\nCSV:", path)

    print(
        "Hronologija: CSV mora biti od najstarijeg "
        "ka najnovijem kolu."
    )

    print(
        f"Validacija: {validation}; "
        f"odvojeni test: {test}"
    )

    for mode in (True, False):
        values = result[mode]

        print(
            f"{labels[mode]}: "
            f"alpha={chosen[mode]:g}, "
            f"prosek={np.mean(values):.4f}, "
            f"pogoci 0..7="
            f"{np.bincount(values, minlength=8).tolist()}"
        )

    print(
        "Kontrola: prethodno izvuceno kolo:",
        f"{np.mean(result['previous']):.4f}"
    )

    p = prepare(history)

    prediction, score, steps = predict(
        p, chosen[True]
    )

    replay = bool(
        np.any(
            np.all(
                history == prediction,
                axis=1
            )
        )
    )

    print(
        f"SLEDECI RED {len(history) + 1}:",
        ",".join(map(str, prediction))
    )

    print(
        f"Skor={score:.10f}; "
        f"koraci={steps}; "
        f"istorijsko ponavljanje={replay}"
    )

    print(
        "Skor nije verovatnoca. Pretraga: "
        "istorijska polazista i dozvoljene"
    )

    print(
        "zamene jednog broja; lokalni optimum, "
        "bez tvrdnje o globalnom."
    )

    print(
        "Ucenje skalarnog izlaznog skora je dodatak "
        "originalnoj logickoj mrezi."
    )

    return prediction


def self_test():
    # Cela tablica istinitosti AND i OR.
    truth = np.array((
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1)
    ), dtype=bool)

    assert np.array_equal(
        neuron(truth, 2),
        (False, False, False, True)
    )

    assert np.array_equal(
        neuron(truth, 1),
        (False, True, True, True)
    )

    # Inhibicija blokira i uz mnogo pobudnih ulaza.
    assert not neuron(
        np.ones((1, 20), dtype=bool),
        1,
        np.ones((1, 1), dtype=bool)
    )[0]

    values = np.repeat(
        np.arange(33)[:, None],
        8,
        axis=1
    )

    bits = gap_bits(
        values
    ).reshape(33, 8, 6)

    assert np.array_equal(
        np.sum(
            bits * (1 << np.arange(6)),
            axis=-1
        ),
        values
    )

    base = np.array((
        (1, 2, 3, 4, 5, 6, 7),
        (4, 12, 18, 28, 35, 36, 37),
        (33, 34, 35, 36, 37, 38, 39),
        (4, 8, 13, 18, 23, 28, 34)
    ))

    g = gaps(base)

    assert np.all(
        g.sum(axis=1) == 32
    )

    assert np.array_equal(
        np.cumsum(
            g[:, :-1] + 1,
            axis=1
        ),
        base
    )

    h = np.tile(base, (32, 1))

    for mode in (True, False):
        p = prepare(h, mode)
        model = fit(p, .1)

        scaled = model[0] * p[2]

        assert np.allclose(
            (
                p[0] + .1 * np.eye(DIM)
            ) @ scaled,
            p[1]
        )

        f = features(g, p[3])

        assert f.shape == (4, DIM)
        assert np.all((f == 0) | (f == 1))

        prediction, score, _ = predict(
            p, .1
        )

        assert prediction == predict(
            prepare(h.copy(), mode),
            .1
        )[0]

        assert (
            len(set(prediction)) == 7
            and 1 <= min(prediction)
            <= max(prediction) <= 39
        )

        tree, radius = support(p[4])

        other = replacements(
            np.array(prediction)
        )

        other = other[
            allowed(
                other,
                tree,
                radius
            )
        ]

        assert (
            scores(
                other, model
            ).max(initial=-np.inf)
            <= score + 1e-9
        )

        if not mode:
            assert np.all(
                f[:, 72:104] == 0
            )

            assert np.all(
                f[:, 118:] == 0
            )

    future = h.copy()
    future[-1] = base[1]

    assert predict(
        prepare(h[:-1]), .1
    )[0] == predict(
        prepare(future[:-1]), .1
    )[0]

    print(
        "Self-test OK: pragovi, inhibicija, "
        "kodiranje, ucenje i hronologija."
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "csv",
        nargs="*",
        help="Podrazumevano sva tri CSV fajla."
    )

    parser.add_argument(
        "--validation",
        type=int,
        default=48
    )

    parser.add_argument(
        "--test",
        type=int,
        default=64
    )

    parser.add_argument(
        "--self-test",
        action="store_true"
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()

    for path in args.csv or DEFAULTS:
        run_file(
            path,
            args.validation,
            args.test
        )


if __name__ == "__main__":
    main()



"""
CSV: /Users/4c/Desktop/GHQ/data/loto7_4700_k81.csv
Hronologija: CSV mora biti od najstarijeg ka najnovijem kolu.
Validacija: 48; odvojeni test: 64
MCCULLOCH-PITTS: LOGIKA I VREMENSKI USLOVI: alpha=0.1, prosek=1.3281, pogoci 0..7=[7, 31, 24, 2, 0, 0, 0, 0]
Kontrola: bez vremenskih logickih uslova: alpha=1, prosek=1.1875, pogoci 0..7=[16, 27, 15, 5, 1, 0, 0, 0]
Kontrola: prethodno izvuceno kolo: 1.2344
SLEDECI RED 4701: 4,5,8,10,19,36,38
Skor=1.7158862108; koraci=7; istorijsko ponavljanje=False
Skor nije verovatnoca. Pretraga: istorijska polazista i dozvoljene
zamene jednog broja; lokalni optimum, bez tvrdnje o globalnom.
Ucenje skalarnog izlaznog skora je dodatak originalnoj logickoj mrezi.



CSV: /Users/4c/Desktop/GHQ/data/loto7_4700_k81_loto_2972.csv
Hronologija: CSV mora biti od najstarijeg ka najnovijem kolu.
Validacija: 48; odvojeni test: 64
MCCULLOCH-PITTS: LOGIKA I VREMENSKI USLOVI: alpha=1, prosek=1.0312, pogoci 0..7=[19, 29, 13, 1, 2, 0, 0, 0]
Kontrola: bez vremenskih logickih uslova: alpha=0.1, prosek=1.4375, pogoci 0..7=[11, 27, 17, 5, 4, 0, 0, 0]
Kontrola: prethodno izvuceno kolo: 1.3281
SLEDECI RED 2973: 17,25,26,27,32,36,38
Skor=1.5437126169; koraci=8; istorijsko ponavljanje=False
Skor nije verovatnoca. Pretraga: istorijska polazista i dozvoljene
zamene jednog broja; lokalni optimum, bez tvrdnje o globalnom.
Ucenje skalarnog izlaznog skora je dodatak originalnoj logickoj mrezi.



CSV: /Users/4c/Desktop/GHQ/data/loto7_4700_k81_loto_plus_1728.csv
Hronologija: CSV mora biti od najstarijeg ka najnovijem kolu.
Validacija: 48; odvojeni test: 64
MCCULLOCH-PITTS: LOGIKA I VREMENSKI USLOVI: alpha=0.1, prosek=1.2812, pogoci 0..7=[18, 19, 18, 9, 0, 0, 0, 0]
Kontrola: bez vremenskih logickih uslova: alpha=0.01, prosek=1.4062, pogoci 0..7=[11, 27, 15, 11, 0, 0, 0, 0]
Kontrola: prethodno izvuceno kolo: 1.2969
SLEDECI RED 1729: 1,2,3,12,29,35,38
Skor=2.0614609975; koraci=6; istorijsko ponavljanje=False
Skor nije verovatnoca. Pretraga: istorijska polazista i dozvoljene
zamene jednog broja; lokalni optimum, bez tvrdnje o globalnom.
Ucenje skalarnog izlaznog skora je dodatak originalnoj logickoj mrezi.
"""



"""
1943, Walter Pitts and Warren McCulloch

Njihov rad „A Logical Calculus of the Ideas Immanent in Nervous Activity” iz 1943. 
U njemu su opisali mrežu binarnih neurona: neuron se aktivira kada pobudni ulazi dostignu prag, dok aktivan inhibitorni ulaz blokira aktivaciju. 
Uključili su vremenska kašnjenja i povratne veze. 

Za Loto ideja: logičku mrežu koja prepoznaje zajedničke uslove u rasporedu razmaka i njihovim prethodnim promenama.

- Opisivati celu kombinaciju kroz osam praznih razmaka.
- Pretvarati odnose razmaka i promene kroz prethodna kola u logičke uslove.
- Povezivati uslove operacijama AND, OR i inhibicijom, uz pamćenje prethodnih stanja.
- Učiti koji aktivirani obrasci doprinose skoru cele kandidatske kombinacije, pa birati jednu konkretnu sedmorku.

Primer obrasca: „razmak A se povećao, razmak B smanjio, a prethodno stanje ispunjava uslov C”. 
Korisna ideja je zajedničko delovanje uslova kroz vreme.
Originalni rad daje logičku konstrukciju mreže; postupak učenja za moje podatke bio bi dodatak koji moram definisati i proveriti. 

Model koristi determinističke pragovne neurone, inhibiciju i prethodna stanja da oceni celu kombinaciju.

Prošle su provere pragovnih neurona, apsolutne inhibicije i bezgubitnog kodiranja razmaka. 


Proverena na 48 validacionih i 64 odvojena testna kola. 
Vremenski logički uslovi pomogli su na zbirnom fajlu, ali nisu doneli dosledno poboljšanje.

CSV	        Prosek pogodaka	Bez vremenskih uslova	Sledeća kombinacija
Zbirni 4700	1,3281	        1,1875	                4,5,8,10,19,36,38
Loto 2972	1,0312	        1,4375	                17,25,26,27,32,36,38
Plus 1728	1,2812	        1,4062	                1,2,3,12,29,35,38

"""
