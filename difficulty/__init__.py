"""Discriminant structure under continuously controlled task difficulty.

The interface is a difficulty axis, not a genomics task. `axis.genomic` supplies
one instance of that axis whose difficulty parameter is an information-theoretic
property of the generating process rather than an ad hoc corruption level; other
instances (a Gaussian axis with analytic Bayes error) are meant to plug in
behind the same interface.

    from difficulty.axis import genomic
    task = genomic.sample(level=3, seed=0)   # -> Task(x, y, difficulty, meta)
"""
