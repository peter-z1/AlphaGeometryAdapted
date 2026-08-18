# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Run DD+AR or AlphaGeometry solver.

Please refer to README.md for detailed instructions.
"""

from __future__ import annotations

import traceback

import sys
import logging
import ddar
import graph as gh
import pretty as pt
import problem as pr
import argparse
import time

DEFINITIONS = None  # contains definitions of construction actions
RULES = None  # contains rules of deductions
LM = None

def natural_language_statement(logical_statement: pr.Dependency) -> str:
    """Convert logical_statement to natural language.

    Args:
      logical_statement: pr.Dependency with .name and .args

    Returns:
      a string of (pseudo) natural language of the predicate for human reader.
    """
    names = [a.name.upper() for a in logical_statement.args]
    names = [(n[0] + '_' + n[1:]) if len(n) > 1 else n for n in names]
    return pt.pretty_nl(logical_statement.name, names)


def proof_step_string(
    proof_step: pr.Dependency, refs: dict[tuple[str, ...], int], last_step: bool
) -> str:
    """Translate proof to natural language.

    Args:
      proof_step: pr.Dependency with .name and .args
      refs: dict(hash: int) to keep track of derived predicates
      last_step: boolean to keep track whether this is the last step.

    Returns:
      a string of (pseudo) natural language of the proof step for human reader.
    """
    premises, [conclusion] = proof_step

    premises_nl = ' & '.join(
        [
            natural_language_statement(p) + ' [{:02}]'.format(refs[p.hashed()])
            for p in premises
        ]
    )

    if not premises:
        premises_nl = 'similarly'

    refs[conclusion.hashed()] = len(refs)

    conclusion_nl = natural_language_statement(conclusion)
    if not last_step:
        conclusion_nl += ' [{:02}]'.format(refs[conclusion.hashed()])

    return f'{premises_nl} \u21d2 {conclusion_nl}'


def write_solution(g: gh.Graph, p: pr.Problem, out_file: str) -> None:
    """Output the solution to out_file.

    Args:
      g: gh.Graph object, containing the proof state.
      p: pr.Problem object, containing the theorem.
      out_file: file to write to, empty string to skip writing to file.
    """
    setup, aux, proof_steps, refs = ddar.get_proof_steps(
        g, p.goal, merge_trivials=False
    )

    solution = '\n=========================='
    solution += '\n * From theorem premises:\n'
    premises_nl = []
    for premises, [points] in setup:
        solution += ' '.join([p.name.upper() for p in points]) + ' '
        if not premises:
            continue
        premises_nl += [
            natural_language_statement(p) + ' [{:02}]'.format(refs[p.hashed()])
            for p in premises
        ]
    solution += ': Points\n' + '\n'.join(premises_nl)

    solution += '\n\n * Auxiliary Constructions:\n'
    aux_premises_nl = []
    for premises, [points] in aux:
        solution += ' '.join([p.name.upper() for p in points]) + ' '
        aux_premises_nl += [
            natural_language_statement(p) + ' [{:02}]'.format(refs[p.hashed()])
            for p in premises
        ]
    solution += ': Points\n' + '\n'.join(aux_premises_nl)

    # some special case where the deduction rule has a well known name.
    r2name = {
        'r32': '(SSS)',
        'r33': '(SAS)',
        'r34': '(Similar Triangles)',
        'r35': '(Similar Triangles)',
        'r36': '(ASA)',
        'r37': '(ASA)',
        'r38': '(Similar Triangles)',
        'r39': '(Similar Triangles)',
        'r40': '(Congruent Triangles)',
        'a00': '(Distance chase)',
        'a01': '(Ratio chase)',
        'a02': '(Angle chase)',
    }

    solution += '\n\n * Proof steps:\n'
    for i, step in enumerate(proof_steps):
        _, [con] = step
        nl = proof_step_string(step, refs, last_step=i == len(proof_steps) - 1)
        rule_name = r2name.get(con.rule_name, '')
        nl = nl.replace('\u21d2', f'{rule_name}\u21d2 ')
        solution += '{:03}. '.format(i + 1) + nl + '\n'

    solution += '==========================\n'
    logging.info(solution)
    if out_file:
        with open(out_file, 'w', encoding='utf8') as f:
            f.write(solution)
        logging.info('Solution written to %s.', out_file)


def get_lm(
    backend: str,
    model_file: str,
    model_revision: str,
    checkpoint: str,
    tokenizer: str,
    adapter: str,
    batch_size: int,
    max_decode_len: int,
    device: str,
    dtype: str,
    load_in_4bit: bool,
    hf_cache_dir: str,
):
    """Construct one of the optional language-model inference backends."""
    global LM
    if LM is not None:
        return LM

    if backend == 'pytorch':
        if not checkpoint or not tokenizer:
            raise ValueError(
                '--checkpoint and --tokenizer are required for '
                '--backend=pytorch'
            )
        from pytorch_lm_inference import (  # pylint: disable=import-outside-toplevel
            PytorchLanguageModelInference,
        )
        LM = PytorchLanguageModelInference(
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_decode_len=max_decode_len,
            device=device,
        )
    elif backend == 'chatllm':
        import lm_inference as chatllm_inference  # pylint: disable=import-outside-toplevel
        LM = chatllm_inference.LanguageModelInference(
            model_file=model_file,
            mode='beam_search',
            batch_size=batch_size,
        )
    elif backend == 'huggingface':
        if not model_file or model_file.startswith(':'):
            raise ValueError(
                '--model must name a Hugging Face base model when '
                '--backend=huggingface'
            )
        from huggingface_lm_inference import (  # pylint: disable=import-outside-toplevel
            HuggingFaceLanguageModelInference,
        )
        LM = HuggingFaceLanguageModelInference(
            model_name=model_file,
            revision=model_revision,
            adapter=adapter,
            batch_size=batch_size,
            max_decode_len=max_decode_len,
            device=device,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            cache_dir=hf_cache_dir,
        )
    else:
        raise ValueError(f'Unknown language-model backend: {backend}')

    return LM

def run_ddar(g: gh.Graph, p: pr.Problem, out_file: str) -> bool:
    """Run DD+AR.

    Args:
      g: gh.Graph object, containing the proof state.
      p: pr.Problem object, containing the problem statement.
      out_file: path to output file if solution is found.

    Returns:
      Boolean, whether DD+AR finishes successfully.
    """
    ddar.solve(g, RULES, p, max_level=1000)

    goal_args = g.names2nodes(p.goal.args)
    if not g.check(p.goal.name, goal_args):
        logging.info('DD+AR failed to solve the problem.')
        return False

    write_solution(g, p, out_file)

    gh.nm.draw(
        g.type2nodes[gh.Point],
        g.type2nodes[gh.Line],
        g.type2nodes[gh.Circle],
        g.type2nodes[gh.Segment],
        save_to=(out_file + '.png' if out_file != '' else None))
    return True


def build_problem_with_retries(
    p: pr.Problem,
    definitions: dict[str, pr.Definition],
    max_attempts: int = 20,
) -> tuple[gh.Graph, list[pr.Dependency]]:
    """Build a numerical proof state without rejecting it on one bad sample.

    Graph.build_problem currently calls exit() when a random realization puts
    points too close together or too far apart. Complex problems encounter
    these harmless sampling failures frequently. Retry with fresh randomness
    before deciding that an LM-proposed construction is unusable.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return gh.Graph.build_problem(p, definitions)
        except SystemExit:
            logging.warning(
                'Numerical build failed on attempt %d/%d; resampling.',
                attempt,
                max_attempts,
            )
    raise RuntimeError(
        f'Unable to build a valid numerical realization after {max_attempts} attempts'
    )


def translate_constrained_to_constructive(
    point: str, name: str, args: list[str]
) -> tuple[str, list[str]]:
    """Translate a predicate from constraint-based to construction-based.

    Args:
      point: str: name of the new point
      name: str: name of the predicate, e.g., perp, para, etc.
      args: list[str]: list of predicate args.

    Returns:
      (name, args): translated to constructive predicate.
    """
    if name in ['T', 'perp']:
        a, b, c, d = args
        if point in [c, d]:
            a, b, c, d = c, d, a, b
        if point == b:
            a, b = b, a
        if point == d:
            c, d = d, c
        if a == c and a == point:
            return 'on_dia', [a, b, d]
        return 'on_tline', [a, b, c, d]

    elif name in ['P', 'para']:
        a, b, c, d = args
        if point in [c, d]:
            a, b, c, d = c, d, a, b
        if point == b:
            a, b = b, a
        return 'on_pline', [a, b, c, d]

    elif name in ['D', 'cong']:
        a, b, c, d = args
        if point in [c, d]:
            a, b, c, d = c, d, a, b
        if point == b:
            a, b = b, a
        if point == d:
            c, d = d, c
        if a == c and a == point:
            return 'on_bline', [a, b, d]
        if b in [c, d]:
            if b == d:
                c, d = d, c  # pylint: disable=unused-variable
            return 'on_circle', [a, b, d]
        return 'eqdistance', [a, b, c, d]

    elif name in ['C', 'coll']:
        a, b, c = args
        if point == b:
            a, b = b, a
        if point == c:
            a, b, c = c, a, b
        return 'on_line', [a, b, c]

    elif name in ['^', 'eqangle']:
        # The constrained LM language describes an angle as two lines, hence
        # ``eqangle`` has eight arguments: AB, CD, EF, GH.  The original
        # inverse adapter below predates that representation and operates on
        # the equivalent six-point form (angle ABC = angle DEF).  Collapse
        # each incident pair of lines before applying the established cases.
        if len(args) == 8:
            a, b, c, d, e, f, g, h = args

            def incident_angle(
                line1: tuple[str, str], line2: tuple[str, str]
            ) -> list[str]:
                common = set(line1).intersection(line2)
                if len(common) != 1:
                    raise ValueError(
                        'eqangle line pairs must have exactly one common point'
                    )
                vertex = next(iter(common))
                first = line1[1] if line1[0] == vertex else line1[0]
                second = line2[1] if line2[0] == vertex else line2[0]
                return [first, vertex, second]

            args = incident_angle((a, b), (c, d)) + incident_angle(
                (e, f), (g, h)
            )
        if len(args) != 6:
            raise ValueError(
                f'eqangle expects 8 line-endpoint arguments, got {len(args)}'
            )

        a, b, c, d, e, f = args

        if point in [d, e, f]:
            a, b, c, d, e, f = d, e, f, a, b, c

        x, b, y, c, d = b, c, e, d, f
        if point == b:
            a, b, c, d = b, a, d, c

        if point == d and x == y:  # x p x b = x c x p
            return 'angle_bisector', [point, b, x, c]

        if point == x:
            return 'eqangle3', [x, a, b, y, c, d]

        return 'on_aline', [a, x, b, c, y, d]

    elif name in ['cyclic', 'O']:
        a, b, c = [x for x in args if x != point]
        return 'on_circum', [point, a, b, c]

    return name, args


def check_valid_args(name: str, args: list[str]) -> bool:
    """Check whether a predicate is grammarically correct.

    Args:
      name: str: name of the predicate
      args: list[str]: args of the predicate

    Returns:
      bool: whether the predicate arg count is valid.
    """
    if name == 'perp':
        if len(args) != 4:
            return False
        a, b, c, d = args
        if len({a, b}) < 2:
            return False
        if len({c, d}) < 2:
            return False
    elif name == 'para':
        if len(args) != 4:
            return False
        a, b, c, d = args
        if len({a, b, c, d}) < 4:
            return False
    elif name == 'cong':
        if len(args) != 4:
            return False
        a, b, c, d = args
        if len({a, b}) < 2:
            return False
        if len({c, d}) < 2:
            return False
    elif name == 'coll':
        if len(args) != 3:
            return False
        a, b, c = args
        if len({a, b, c}) < 3:
            return False
    elif name == 'cyclic':
        if len(args) != 4:
            return False
        a, b, c, d = args
        if len({a, b, c, d}) < 4:
            return False
    elif name == 'eqangle':
        if len(args) != 8:
            return False
        a, b, c, d, e, f, g, h = args
        if len({a, b, c, d}) < 3:
            return False
        if len({e, f, g, h}) < 3:
            return False
    return True


def try_translate_constrained_to_construct(string: str, g: gh.Graph) -> str:
    """Whether a string of aux construction can be constructed.

    Args:
      string: str: the string describing aux construction.
      g: gh.Graph: the current proof state.

    Returns:
      str: whether this construction is valid. If not, starts with "ERROR:".
    """
    string = string.strip()
    if not string:
        return 'ERROR: empty construction'

    if string[-1] != ';':
        return 'ERROR: must end with ;'

    parts = string.split(' : ')
    if len(parts) != 2:
        return 'ERROR: too many `:` separated parts'

    head, prem_str = string.split(' : ')
    point = head.strip()

    if len(point) != 1 or point == ' ':
        return f'ERROR: invalid point name {point}'

    existing_points = [p.name for p in g.all_points()]
    if point in existing_points:
        return f'ERROR: point {point} already exists.'

    prem_toks = prem_str.split()[:-1]  # remove the EOS ' ;'
    prems = [[]]

    for i, tok in enumerate(prem_toks):
        if tok.isdigit():
            if i < len(prem_toks) - 1:
                prems.append([])
        else:
            prems[-1].append(tok)

    if len(prems) > 2:
        return 'ERROR: there cannot be more than two predicates.'

    clause_txt = point + ' = '
    constructions = []
    repeated_output_aline = False

    for prem in prems:
        name, *args = prem

        if point not in args:
            return f'ERROR: {point} not found in predicate args.'

        if not check_valid_args(pt.map_symbol(name), args):
            return 'ERROR: Invalid predicate ' + name + ' ' + ' '.join(args)

        for a in args:
            if a != point and a not in existing_points:
                return f'ERROR: point {a} does not exist.'

        try:
            name, args = translate_constrained_to_constructive(
                point, name, args)
        except:  # pylint: disable=bare-except
            return 'ERROR: Invalid predicate ' + name + ' ' + ' '.join(args)

        if name == 'on_aline':
            if args.count(point) > 1:
                # ``on_bline`` emits a congruence plus a redundant angle
                # equality.  The congruence already translates to the exact
                # perpendicular-bisector locus, while the generic angle
                # inverse cannot express its repeated output safely.
                repeated_output_aline = True
                continue

        constructions += [name + ' ' + ' '.join(args)]

    if repeated_output_aline and not any(
        construction.startswith('on_bline ') for construction in constructions
    ):
        return f'ERROR: on_aline involves twice {point}'

    clause_txt += ', '.join(constructions)
    clause = pr.Clause.from_txt(clause_txt)

    try:
        g.copy().add_clause(clause, 0, DEFINITIONS)
    except:  # pylint: disable=bare-except
        return 'ERROR: ' + traceback.format_exc()

    return clause_txt


def try_translate_constrained_sequence_to_construct(
    string: str, g: gh.Graph
) -> str:
    """Validate a semicolon-delimited sequence of auxiliary point clauses.

    Training targets can contain several dependent point clauses.  Each clause
    must therefore be checked against a graph containing the earlier clauses,
    rather than passing the whole completion to the one-clause translator.
    """
    string = string.strip()
    if not string:
        return 'ERROR: empty construction'
    if not string.endswith(';'):
        return 'ERROR: must end with ;'

    segments = [segment.strip() for segment in string.split(';') if segment.strip()]
    if not segments:
        return 'ERROR: empty construction'

    graph = g.copy()
    translations = []
    for index, segment in enumerate(segments):
        translation = try_translate_constrained_to_construct(segment + ' ;', graph)
        if translation.startswith('ERROR:'):
            return f'ERROR: clause {index + 1}: {translation[7:].strip()}'
        try:
            clause = pr.Clause.from_txt(translation)
            graph.add_clause(clause, index, DEFINITIONS)
            # Graph.copy() rebuilds from build_def, so retain each accepted
            # incremental clause there before validating the next dependency.
            graph.build_def[0].clauses.append(clause)
        except Exception:  # pylint: disable=broad-except
            return f'ERROR: clause {index + 1}: ' + traceback.format_exc()
        translations.append(translation)

    return '; '.join(translations)


def insert_aux_to_premise(pstring: str, auxstring: str) -> str:
    """Insert auxiliary constructs from proof to premise.

    Args:
      pstring: str: describing the problem to solve.
      auxstring: str: describing the auxiliar construction.

    Returns:
      str: new pstring with auxstring inserted before the conclusion.
    """
    setup, goal = pstring.split(' ? ')
    return setup + '; ' + auxstring + ' ? ' + goal


class BeamQueue:
    """Keep only the top k objects according to their values."""

    def __init__(self, max_size: int = 512):
        self.queue = []
        self.max_size = max_size

    def add(self, node: object, val: float) -> None:
        """Add a new node to this queue."""

        if len(self.queue) < self.max_size:
            self.queue.append((val, node))
            return

        # Find the minimum node:
        min_idx, (min_val, _) = min(enumerate(self.queue), key=lambda x: x[1])

        # replace it if the new node has higher value.
        if val > min_val:
            self.queue[min_idx] = (val, node)

    def __iter__(self):
        for val, node in self.queue:
            yield val, node

    def __len__(self) -> int:
        return len(self.queue)


def run_alphageometry(
    model: object,
    p: pr.Problem,
    search_depth: int,
    beam_size: int,
    out_file: str,
    search_stats: dict[str, int] | None = None,
) -> bool:
    """Simplified code to run AlphaGeometry proof search.

    We removed all optimizations that are infrastructure-dependent, e.g.
    parallelized model inference on multi GPUs,
    parallelized DD+AR on multiple CPUs,
    parallel execution of LM and DD+AR,
    shared pool of CPU workers across different problems, etc.

    Many other speed optimizations and abstractions are also removed to
    better present the core structure of the proof search.

    Args:
      model: Interface with inference-related endpoints to JAX's model.
      p: pr.Problem object describing the problem to solve.
      search_depth: max proof search depth.
      beam_size: beam size of the proof search.
      out_file: path to output file if solution is found.
      search_stats: optional mutable counter map for candidate diagnostics.

    Returns:
      boolean of whether this is solved.
    """
    if search_stats is None:
        search_stats = {}

    def increment(name: str, amount: int = 1) -> None:
        search_stats[name] = search_stats.get(name, 0) + amount

    # translate the problem to a string of grammar that the LM is trained on.
    string = p.setup_str_from_problem(DEFINITIONS)
    # special tokens prompting the LM to generate auxiliary points.
    string += ' {F1} x00'
    # the graph to represent the proof state.
    g, _ = build_problem_with_retries(p, DEFINITIONS)

    # First we run the symbolic engine DD+AR:
    if run_ddar(g, p, out_file):
        return True

    # beam search for the proof
    # each node in the search tree is a 3-tuple:
    # (<graph representation of proof state>,
    #  <string for LM to decode from>,
    #  <original problem string>)
    beam_queue = BeamQueue(max_size=beam_size)
    # originally the beam search tree starts with a single node (a 3-tuple):
    beam_queue.add(
        # value of the root node is simply 0.
        node=(g, string, p.txt()), val=0.0
    )

    for depth in range(search_depth):
        logging.info(
            'Depth %s. There are %i nodes to expand:', depth, len(beam_queue)
        )
        for _, (_, string, _) in beam_queue:
            logging.info(string)

        new_queue = BeamQueue(max_size=beam_size)  # to replace beam_queue.

        for prev_score, (g, string, pstring) in beam_queue:
            logging.info('Decoding from %s', string)
            outputs = model.beam_decode(string, eos_tokens=[';'])
            increment('decode_calls')
            increment('decoded_candidates', len(outputs['seqs_str']))

            # translate lm output to the constructive language.
            # so that we can update the graph representing proof states:
            translations = [
                try_translate_constrained_to_construct(o, g)
                for o in outputs['seqs_str']
            ]

            # couple the lm outputs with its translations
            candidates = zip(outputs['seqs_str'],
                             translations, outputs['scores'])

            # sort candidates by scores
            candidates = sorted(list(candidates), key=lambda x: x[2], reverse=True)

            for lm_out, translation, score in candidates:
                logging.info('LM output (score=%f): "%s"', score, lm_out)
                logging.info('Translation: "%s"\n', translation)

                if translation.startswith('ERROR:'):
                    # the construction is invalid.
                    increment('translation_rejections')
                    continue

                # Update the constructive statement of the problem with the aux point:
                candidate_pstring = insert_aux_to_premise(pstring, translation)

                logging.info('Solving: "%s"', candidate_pstring)
                try:
                    p_new = pr.Problem.from_txt(candidate_pstring)
                except Exception as exc:  # Candidate failure is not a search failure.
                    increment('candidate_parse_rejections')
                    logging.warning(
                        'Skipping unparsable auxiliary candidate: %s: %s',
                        type(exc).__name__, exc,
                    )
                    continue

                # This is the new proof state graph representation:
                try:
                    g_new, _ = build_problem_with_retries(
                        p_new, DEFINITIONS, max_attempts=20)
                except TimeoutError:
                    raise
                except (Exception, SystemExit) as exc:
                    increment('candidate_build_rejections')
                    logging.warning(
                        'Skipping auxiliary candidate that cannot be built: '
                        '%s: %s', type(exc).__name__, exc,
                    )
                    continue

                try:
                    solved = run_ddar(g_new, p_new, out_file)
                except TimeoutError:
                    raise
                except Exception as exc:  # Candidate failure is not a search failure.
                    increment('candidate_reasoning_rejections')
                    logging.warning(
                        'Skipping auxiliary candidate rejected by DD+AR: %s: %s',
                        type(exc).__name__, exc,
                    )
                    continue

                increment('accepted_candidates')
                if solved:
                    logging.info('Solved.')
                    return True

                # Add the candidate to the beam queue.
                new_queue.add(
                    # The string for the new node is old_string + lm output +
                    # the special token asking for a new auxiliary point ' x00':
                    node=(g_new, string + ' ' + lm_out + \
                          ' x00', candidate_pstring),
                    # the score of each node is sum of score of all nodes
                    # on the path to itself. For beam search, there is no need to
                    # normalize according to path length because all nodes in beam
                    # is of the same path length.
                    val=prev_score + score,
                )
                # Note that the queue only maintain at most beam_size nodes
                # so this new node might possibly be dropped depending on its value.

        # replace the old queue with new queue before the new proof search depth.
        beam_queue = new_queue

    return False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--problems_file',
        type=str,
        default='examples/imo_ag_30.txt',
        help='text file contains the problem strings. See imo_ag_30.txt for example.')

    parser.add_argument(
        '--problem_name',
        type=str,
        default='imo_2000_p1',
        help='name of the problem to solve, must be in the problem_file.')

    parser.add_argument(
        '--mode',
        type=str,
        default='ddar',
        help='either `ddar` (DD+AR) or `alphageometry`.')

    parser.add_argument(
        '--model',
        type=str,
        default=':alphageometry-lm',
        help='ChatLLM model identifier or Hugging Face base-model name.')

    parser.add_argument(
        '--backend',
        choices=('pytorch', 'chatllm', 'huggingface'),
        default='pytorch',
        help='language-model inference backend (default: pytorch).')

    parser.add_argument(
        '--checkpoint',
        type=str,
        default='',
        help='PyTorch reconstruction checkpoint.')

    parser.add_argument(
        '--tokenizer',
        type=str,
        default='',
        help='SentencePiece tokenizer used by the PyTorch checkpoint.')

    parser.add_argument(
        '--adapter',
        type=str,
        default='',
        help='optional PEFT/LoRA adapter for the Hugging Face backend.')

    parser.add_argument(
        '--model_revision',
        type=str,
        default='',
        help='optional Hugging Face model commit or tag.')

    parser.add_argument(
        '--dtype',
        choices=('bfloat16', 'float16', 'float32'),
        default='bfloat16',
        help='floating-point type used by the Hugging Face backend.')

    parser.add_argument(
        '--load_in_4bit',
        action='store_true',
        help='load Hugging Face base weights in four-bit form.')

    parser.add_argument(
        '--hf_cache_dir',
        type=str,
        default='',
        help='optional Hugging Face model/tokenizer cache directory.')

    parser.add_argument(
        '--device',
        choices=('cpu', 'cuda'),
        default='cuda',
        help='inference device; Hugging Face CUDA mode requires a visible GPU.')

    parser.add_argument(
        '--max_decode_len',
        type=int,
        default=32,
        help='maximum generated tokens per auxiliary construction.')

    parser.add_argument(
        '--defs_file',
        type=str,
        default='data/defs.txt',
        help='definitions of available constructions to state a problem.')

    parser.add_argument(
        '--rules_file',
        type=str,
        default='data/rules.txt',
        help='list of deduction rules used by DD.')

    parser.add_argument(
        '--out_file',
        type=str,
        default='',
        help='path to the solution output file.')

    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='batch size (i.e. beam size) of LLM.')

    parser.add_argument(
        '--beam_size',
        type=int,
        default=2,
        help='beam size of the proof search.')

    parser.add_argument(
        '--search_depth',
        type=int,
        default=2,
        help='search depth of the proof search.')

    return parser.parse_args()

def main(FLAGS):
    global DEFINITIONS
    global RULES

    # definitions of terms used in our domain-specific language.
    DEFINITIONS = pr.Definition.from_txt_file(FLAGS.defs_file, to_dict=True)
    # load inference rules used in DD.
    RULES = pr.Theorem.from_txt_file(FLAGS.rules_file, to_dict=True)

    # when using the language model,
    # point names will be renamed to alphabetical a, b, c, d, e, ...
    # instead of staying with their original names,
    # in order to match the synthetic training data generation.
    need_rename = FLAGS.mode != 'ddar'

    logging.info(f"{FLAGS}")

    # load problems from the problems_file,
    problems = pr.Problem.from_txt_file(
        FLAGS.problems_file, to_dict=True, translate=need_rename
    )

    if FLAGS.problem_name not in problems:
        raise ValueError(
            f'Problem name `{FLAGS.problem_name}` '
            + f'not found in `{FLAGS.problems_file}`'
        )

    this_problem = problems[FLAGS.problem_name]

    if FLAGS.mode == 'ddar':
        g, _ = gh.Graph.build_problem(this_problem, DEFINITIONS)
        run_ddar(g, this_problem, FLAGS.out_file)

    elif FLAGS.mode == 'alphageometry':
        model = get_lm(
            backend=FLAGS.backend,
            model_file=FLAGS.model,
            model_revision=FLAGS.model_revision,
            checkpoint=FLAGS.checkpoint,
            tokenizer=FLAGS.tokenizer,
            adapter=FLAGS.adapter,
            batch_size=FLAGS.batch_size,
            max_decode_len=FLAGS.max_decode_len,
            device=FLAGS.device,
            dtype=FLAGS.dtype,
            load_in_4bit=FLAGS.load_in_4bit,
            hf_cache_dir=FLAGS.hf_cache_dir,
        )
        run_alphageometry(
            model,
            this_problem,
            FLAGS.search_depth,
            FLAGS.beam_size,
            FLAGS.out_file,
        )

    else:
        raise ValueError(f'Unknown FLAGS.mode: {FLAGS.mode}')


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    FLAGS = parse_args()

    start_time = time.perf_counter()
    if FLAGS.problem_name != '':
        main(FLAGS)
    else:
        out_file = FLAGS.out_file
        for name in pr.Problem.from_txt_file(FLAGS.problems_file, to_dict=True).keys():
            if name.startswith('#'): continue
            FLAGS.problem_name = name
            FLAGS.out_file = f"{out_file}/{name}.txt" if out_file != '' else ''
            try:
                main(FLAGS)
            except Exception as e:
                print(f"exception occurred on {name}:\n{e}")

    elapsed_time = time.perf_counter() - start_time
    logging.info(f"Total time = {elapsed_time:.4f} seconds")
