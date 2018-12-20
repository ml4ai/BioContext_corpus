# Zechy Wong
# January 2017
# Given data from multiple annotators, calculate and display general
# overlap/agreement counts.

from __future__ import division, print_function

import collections
import csv
import itertools
import os
import pprint
import re
from argparse import Namespace

from fleiss_kappa import computeKappa

# Module-level config
# Data folders should include the following files for each paper:
# <paper_id>_contexts.tsv, <paper_id>_events.tsv, <paper_id>_reach.tsv
context_suffix = "_contexts.tsv"
event_suffix = "_events.tsv"
reach_suffix = "_reach.tsv"
grounding_suffix = "_groundings.tsv"

# Global queries against the grounding->free_text dictionary
# all_groundings.tsv should be in the root data folder
all_groundings_file = "all_groundings.tsv"
all_groundings = {}


def main():
    """
    Top-level logic
    """
    # =-=-=-=
    # Config
    # =-=-=-=
    data_path = os.path.normpath("../corpus_data")
    # Annotator IDs -- Should also be sub-folders in the data folder
    annotators = ["annotator1", "annotator2", "annotator3"]
    # annotator3 seems to have a lot of unique events:
    # Tuple keys for count dictionaries; sorted in the following order when
    # displayed
    annotator_order = {
        'reach': 0,
        'annotator1':   1,
        'annotator2':  2,
        'annotator3': 3,
        'human': 4  # Grouped case
    }
    # --- The following config options will be set per-iteration to show all
    # the combinations we are interested in

    # # Should event overlap calculations check against Reach events?
    # events_include_reach = True

    # # How should human annotators be grouped together for comparing event
    # # overlap counts with Reach?
    # #   'none' -- Each annotator is considered separately.
    # #   'intersection' -- Only events picked out by all the human annotators
    # #   will be considered.
    # #   'union' -- All the events picked out by human annotators will be
    # #   considered.
    # group_humans = "intersection"

    # =-=-=
    # Init
    # =-=-=
    print("Calculating counts for the following annotators:\n"
          "{}"
          "".format(annotators))

    paper_set = get_shared_papers(data_path, annotators)
    print("Considering {} shared papers.".format(len(paper_set)))

    init_grounding_db(data_path)

    # =-=-=-=-=-=-=-=
    # Manual Events
    # =-=-=-=-=-=-=-=
    # Step 1: Calculate overlaps with Reach
    # We can calculate: Reach only, Annotator + Reach only, Annotator only
    #                   Reach with all annotators, Inter-annotator only
    #
    # We will show all the combinations we are interested in.
    config = Namespace()
    config.data_path = data_path
    config.annotators = annotators
    config.annotator_order = annotator_order
    config.paper_set = paper_set
    # config.show_preamble
    # config.show_paper_event_counts
    # config.events_include_reach
    # config.group_humans

    # Combo 1: Individual human annotators
    config.show_preamble = True
    config.show_paper_event_counts = False
    config.events_include_reach = False
    config.group_humans = "none"
    process_manual_events(config)

    # Combo 2: Individual human annotators with Reach
    config.show_preamble = False
    config.show_paper_event_counts = False
    config.events_include_reach = True
    config.group_humans = "none"
    process_manual_events(config)

    # Combo 3: Grouped human annotators with Reach (intersection)
    config.show_preamble = False
    config.show_paper_event_counts = False
    config.events_include_reach = True
    config.group_humans = "intersection"
    process_manual_events(config)

    # Combo 4: Grouped human annotators with Reach (union)
    config.show_preamble = False
    config.show_paper_event_counts = False
    config.events_include_reach = True
    config.group_humans = "union"
    process_manual_events(config)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Context Association Kappas
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=
    # A (Fleiss') Kappa score will be calculated for each context type that we
    # observe in the data.
    config = Namespace()
    config.data_path = data_path
    config.annotators = annotators
    config.annotator_order = annotator_order
    config.paper_set = paper_set

    # If True, will show cases where the context in question was never
    # associated with an event OR was associated with all events
    config.show_edge_cases = False

    # If True, will display the list of context kappa scores in a more
    # compact fashion
    config.compact_score_list = False

    # If True, will also show all the free-text strings associated with a given
    # grounding ID
    config.show_grounding_texts = True
    process_associations_kappa(config)


def init_grounding_db(data_path):
    """
    Prepares the grounding -> free_text dictionary for queries
    :return:
    """
    path = os.path.join(data_path, all_groundings_file)
    with open(path, 'rb') as fp:
        tsv = csv.reader(fp, delimiter='\t')
        for row in tsv:
            grounding_id = row[0]
            free_texts = row[1]
            all_groundings[grounding_id] = free_texts


def get_free_texts(grounding_id):
    """
    Gets the free-text strings associated with the given grounding ID
    :param grounding_id:
    :return:
    """
    return all_groundings[grounding_id]


def get_shared_papers(data_path, annotators):
    """
    Enumerate the files in the folders, figuring out which paper IDs
    are shared between the annotators
    :return: Set
    """
    assert len(annotators) > 1

    paper_set = set()
    pattern = re.compile(r"(.*)({}|{}|{})"
                         r"".format(context_suffix, event_suffix, reach_suffix))
    for filename in os.listdir(os.path.join(data_path, annotators[0])):
        match = pattern.match(filename)
        if match is not None:
            paper_set.add(match.group(1))

    # Sanity checks: Are all the files present for all annotators?
    for paper_id in paper_set:
        filenames = [paper_id + suffix
                     for suffix in
                     [context_suffix, event_suffix, reach_suffix]]

        mark_remove = False

        for annotator in annotators:
            for filename in filenames:
                if not os.path.isfile(os.path.join(data_path,
                                                   annotator,
                                                   filename)):
                    print("[WARN] Missing file for '{}'; removing from set of "
                          "papers to consider. ({})"
                          "".format(annotator, os.path.join(annotator,
                                                            filename)))
                    mark_remove = True

        if mark_remove:
            paper_set.remove(paper_id)

    return paper_set


def get_event_sets(data_path, annotators, paper_id):
    """
    Given some list of annotators, and some specified paper_id, returns lists of
    the events contained in the corresponding _events.tsv file, represented
    as Namespaces with appropriate properties.
    Format:
    <line>  <interval>  <associations>
    """
    # Will hold both lists of return values when we're done
    return_lists = []

    for annotator in annotators:
        return_list = []
        path = os.path.join(data_path,
                            annotator,
                            paper_id + event_suffix)
        with open(path, 'rb') as fp:
            tsv = csv.reader(fp, delimiter='\t')
            for row in tsv:
                this_event = Namespace()

                this_event.line_num = int(row[0])

                intervals = row[1].split("-")
                this_event.start = int(intervals[0])
                this_event.end = int(intervals[1])

                this_event.associations = row[2].split(",")

                return_list.append(this_event)

        return_lists.append(return_list)

    # Should hold the parsed events for annotator 1 and 2
    return return_lists


def get_reach_events(data_path, annotators, paper_id, by_annotator=False):
    """
    Does the same thing as get_event_sets, but for Reach events.
    The data for each annotator excludes false positives; we will only use
    events that were not marked as FPs by any of them.
    By default, only returns the Reach events as seen by the first annotator
    (i.e., with only the first annotator's associations)
    :param by_annotator: If True, overrides the default behaviour to return a
    dictionary of common Reach events key-ed by annotator ID
    """
    # Read the non-FP Reach events for each annotator
    reach_events = {}

    for annotator in annotators:
        reach_events[annotator] = []

        path = os.path.join(data_path,
                            annotator,
                            paper_id + reach_suffix)
        with open(path, 'rb') as fp:
            tsv = csv.reader(fp, delimiter='\t')
            for row in tsv:
                this_event = Namespace()

                this_event.line_num = int(row[0])

                intervals = row[1].split("-")
                this_event.start = int(intervals[0])
                this_event.end = int(intervals[1])

                if row[2] == "":
                    this_event.associations = []
                else:
                    this_event.associations = row[2].split(",")

                reach_events[annotator].append(this_event)

    # Take a copy of the first annotator's events; remove any events that are
    # not in the other annotators' data (i.e., they were marked as FPs)
    return_events = []
    # for annotator in annotators:
    #     print("'{}' has {} Reach events"
    #           "".format(annotator, len(reach_events[annotator])))

    for event_1 in reach_events[annotators[0]][:]:

        # Will be set to False if *any* annotators are missing this event
        found_all_matches = True

        for annotator_2 in [x for x in annotators
                            if x != annotators[0]]:

            # Will be set to True if we find a match for the event for *this*
            # annotator
            found_match = False

            for event_2 in reach_events[annotator_2]:
                # Match line_num, start, and end.
                # `associations` does *not* need to match.
                if has_same_event_interval(event_1, event_2):
                    found_match = True
                    break

            if not found_match:
                # print("No match for event {}({}:{}-{}) by '{}', marked as FP "
                #       "by '{}'."
                #       "".format(paper_id, event_1.line_num, event_1.start,
                #                 event_1.end, annotators[0], annotator_2))
                found_all_matches = False

        if found_all_matches:
            return_events.append(event_1)

    # In the default case (just for the counts), we're done
    if not by_annotator:
        return return_events

    # If we're still here, pack up the Reach events for each annotator into a
    # dictionary
    return_dict = {}
    return_dict[annotators[0]] = return_events

    # Do this by re-checking each annotator against the common set we saved
    # under the first annotator ID
    # (We couldn't generate the intersection on the first pass through)
    for annotator in [x for x in annotators
                      if x != annotators[0]]:

        return_dict[annotator] = []

        for this_event in reach_events[annotator]:

            # Check `this_event` against the events in the common set
            found_match = False

            for common_event in return_events:
                if has_same_event_interval(this_event, common_event):
                    found_match = True
                    break

            if found_match:
                return_dict[annotator].append(this_event)

    # Sanity check?
    test_lists = return_dict.values()
    test_length = len(test_lists.pop())
    for test_list in test_lists:
        assert len(test_list) == test_length

    return return_dict


def get_event_overlaps(annotators, events, annotator_order):
    """
    Given a list of annotator IDs and a corresponding dictionary of event
    lists (where each event list is keyed to a corresponding annotator ID),
    return a dictionary of lists of overlapping events, where the keys are
    various sorted combinations of annotator IDs
    :param annotators:
    :param events:
    :param annotator_order: Dictionary specifying the annotator sort order
    for the results returned
    :return:
    """

    # For sorting multiple annotator IDs consistently
    def annotator_sort(annotator):
        return annotator_order[annotator]

    # We will return key-indexed lists of overlapping events (that can be
    # counted by the caller)
    return_overlaps = collections.defaultdict(list)

    # Count every event once; the sum of the final counts in the overlap
    # dictionaries should be the total number of events.
    # Seems a little inefficient, but not sure it can be done better.
    for annotator_1 in annotators:
        for event_1 in events[annotator_1]:
            # event_1 is the event in question for this iteration
            marked_by = [annotator_1]
            for annotator_2 in [x for x in annotators
                                if x != annotator_1]:
                for event_2 in events[annotator_2]:
                    # Make sure we're on the same line
                    if event_1.line_num != event_2.line_num:
                        continue

                    if (event_1.start <= event_2.start <= event_1.end) or \
                            (event_2.start <= event_1.start <= event_2.end):
                        # We found an overlap with this annotator.
                        # Report it and move on to the next comparison
                        # event.
                        # -- Debugging info
                        # print("Overlap: [{}] {}:{}-{}, [{}] {}:{}-{}"
                        #       "".format(annotator_1,
                        #                 event_1.line_num,
                        #                 event_1.start,
                        #                 event_1.end,
                        #                 annotator_2,
                        #                 event_2.line_num,
                        #                 event_2.start,
                        #                 event_2.end))
                        marked_by.append(annotator_2)
                        # Noted this second annotator's overlap; continue
                        # checking other annotators
                        break

            marked_by = tuple(sorted(marked_by, key=annotator_sort))

            # Record the overlap
            return_overlaps[marked_by].append(event_1)

    return return_overlaps


def get_available_groundings(data_path, annotators, paper_id):
    """
    With the given paper_id, gets the groundings that were available to all
    the annotators
    :param data_path:
    :param annotators:
    :param paper_id:
    :return:
    """
    # Keep the groundings from each annotator in sets first, then return its
    # intersection as a list
    grounding_sets = {}
    for annotator in annotators:
        grounding_sets[annotator] = set()
        path = os.path.join(data_path,
                            annotator,
                            paper_id + grounding_suffix)
        with open(path) as fp:
            for grounding in fp:
                grounding_sets[annotator].add(grounding.strip())

    final_sets = grounding_sets.values()[:]
    return_set = final_sets.pop()
    for final_set in final_sets:
        return_set = return_set.intersection(final_set)

    return sorted(list(return_set))


def print_overlaps(overlap_keys, overlap_data):
    """
    Prints a formatted representation of the given overlap counts.
    Expects a dictionary with tuple keys representing the IDs of the
    annotators being tracked.
    :param overlap_keys: Specify in the order that columns should be in
    :param overlap_data:
    :return:
    """
    left_margin = 2
    h_padding = 1

    # Calculate each column's content width, excluding padding and border
    col_widths = []
    # Also calculate the total number of header rows to prepare
    header_rows = 0
    for key in overlap_keys:
        header_rows = max(header_rows, len(key))

        # We're adding the word '(only)' for clarity to some of the rows
        this_width = len("(only)")

        for line in key:
            this_width = max(this_width, len(line))

        col_widths.append(this_width)

    # Modify the column titles as necessary
    col_titles = overlap_keys[:]
    for i in range(len(col_titles)):
        col_titles[i] = list(col_titles[i])
        for j in range(len(col_titles[i]) - 1):
            col_titles[i][j] += ","

        if len(col_titles[i]) < header_rows:
            col_titles[i].append("(only)")

    # Prepare display
    row_pre = "{:<{}}".format("", left_margin)
    row_post = "\n"
    table_str = ""

    # Header
    table_str += row_pre
    table_str += "+"
    for width in col_widths:
        table_str += ("{:{}<{}}+".format("", "-", width + 2 * h_padding))
    table_str += row_post

    for row in range(header_rows):
        table_str += row_pre
        table_str += "|"
        for col_num in range(len(col_widths)):
            # The tuple keys will be split into elements and bottom-aligned
            # in the header
            this_row = ""
            key_idx = len(col_titles[col_num]) + row - header_rows
            if key_idx >= 0:
                this_row = col_titles[col_num][key_idx]
            table_str += ("{:^{}}|"
                          .format(this_row,
                                  col_widths[col_num] + 2 * h_padding))
        table_str += row_post

    table_str += row_pre
    table_str += "+"
    for width in col_widths:
        table_str += ("{:{}<{}}+".format("", "-", width + 2 * h_padding))
    table_str += row_post

    # Counts
    table_str += row_pre
    table_str += "|"
    for col_num in range(len(col_widths)):
        table_str += ("{:^{}}|"
                      .format(overlap_data[overlap_keys[col_num]],
                              col_widths[col_num] + 2 * h_padding))
    table_str += row_post

    table_str += row_pre
    table_str += "+"
    for width in col_widths:
        table_str += ("{:{}<{}}+".format("", "-", width + 2 * h_padding))
    table_str += row_post

    print(table_str, end='')


def process_manual_events(config):
    """
    Calculate and display event overlap counts against Reach with various
    configuration options
    :return:
    """
    if config.show_preamble:
        print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        print("Overlap in Marked Events")
        print("=========================")

    # ---- Pull from `config` variable
    # config.show_preamble
    # config.show_paper_event_counts
    data_path = config.data_path
    annotators = config.annotators
    annotator_order = config.annotator_order
    paper_set = config.paper_set
    events_include_reach = config.events_include_reach
    group_humans = config.group_humans
    # ----

    # Set up list of annotators and handle grouping cases for human
    # annotators (just for the dictionary keys)
    assert (group_humans in ["none", "intersection", "union"])
    if group_humans == "intersection" or group_humans == "union":
        # We will check which events were picked out by all human annotators
        paper_annotators = ['human']
    else:
        # No action needed if humans are not grouped
        paper_annotators = annotators[:]

    # Add Reach as an annotator where necessary
    if events_include_reach or group_humans != "none":
        paper_annotators += ['reach']

    # Pre-define the combinations we are interested in, so that we know if
    # the counts turn up 0
    def annotator_sort(annotator):
        return annotator_order[annotator]

    overlap_keys = []
    for i in range(len(paper_annotators)):
        overlap_keys += \
            itertools.combinations(sorted(paper_annotators, key=annotator_sort),
                                   i + 1)

    event_overlaps = {}
    for key in overlap_keys:
        event_overlaps[key] = 0

    # Key to use for checking Jaccard similarity intersection -- The events
    # picked out by all the available annotators
    jaccard_intersection_key = tuple(sorted(paper_annotators,
                                            key=annotator_sort))

    # --- Loop over all the papers in the comparison set ---
    for paper_id in paper_set:
        manual_events = get_event_sets(data_path, annotators, paper_id)
        reach_events = get_reach_events(data_path, annotators, paper_id)

        paper_events = {
            'reach': reach_events
        }

        # Match manual_events lists to annotators
        for x in range(len(manual_events)):
            paper_events[annotators[x]] = manual_events[x]

        if config.show_paper_event_counts:
            print("Paper: {}. Events: {}"
                  "".format(paper_id,
                            sum([len(x) for x in paper_events.values()])))

        # Non-grouped case
        if group_humans == "none":
            raw_overlaps = get_event_overlaps(paper_annotators, paper_events,
                                              annotator_order)
        elif group_humans == "intersection":
            # Pass 1 comparison: Find the events marked by all humans
            # (`annotators` lists all human annotators; `paper_annotators`
            # has the combined `humans` key)
            raw_overlaps = get_event_overlaps(annotators, paper_events,
                                              annotator_order)
            intersect_key = tuple(sorted(annotators, key=annotator_sort))
            human_events = raw_overlaps[intersect_key]

            # Pass 2: Compare those with Reach
            paper_events['human'] = human_events
            raw_overlaps = get_event_overlaps(paper_annotators, paper_events,
                                              annotator_order)
        elif group_humans == "union":
            paper_events['human'] = []
            for annotator in annotators:
                paper_events['human'] += paper_events[annotator]
            raw_overlaps = get_event_overlaps(paper_annotators, paper_events,
                                              annotator_order)

        paper_overlaps = {}
        for key in overlap_keys:
            paper_overlaps[key] = 0

        for combo in raw_overlaps:
            paper_overlaps[combo] = len(raw_overlaps[combo])
            event_overlaps[combo] += len(raw_overlaps[combo])

    print()

    if group_humans != "none":
        print("Grouping human-annotated events ({})".format(group_humans))
    print_overlaps(overlap_keys, event_overlaps)
    total_count = sum(event_overlaps.values())
    jaccard_count = event_overlaps[jaccard_intersection_key]
    print("  Total Events: {}. Jaccard similarity with intersection {}: "
          "{:.03f} "
          "".format(total_count,
                    jaccard_intersection_key, jaccard_count / total_count))


def process_associations_kappa(config):
    """
    Uses the implementation of Fleiss' Kappa at
    https://en.wikibooks.org/wiki/Algorithm_Implementation/Statistics/Fleiss
    %27_kappa#Python
    Raters (n) => Annotators
    Subjects (N) => Events
    Categories (k) => Binary: 0 if not associated, 1 if associated.
    One Kappa value computed per context type.

    Also tries to calculate a meaningful Jaccard similarity score
    :return:
    """
    # [Sample matrix]
    # 1 Row per event.
    # 2 Columns: not associated, associated
    # Cells contain the number of annotators which made that assignment
    # (Each row's values should add up to the total number of annotators)

    # mat = \
    #     [
    #         [1, 2],
    #         [2, 1],
    #         [0, 3],
    #         [0, 3]
    #     ]
    # computeKappa(mat)

    print()
    print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
    print("Fleiss' Kappa - By Context")
    print("===========================")

    # ---- Pull from `config` variable
    data_path = config.data_path
    annotators = config.annotators
    annotator_order = config.annotator_order
    paper_set = config.paper_set
    show_edge_cases = config.show_edge_cases
    compact_score_list = config.compact_score_list
    # ----

    # ----------
    # Dictionary of matrices like the one described above; keys are the
    # context types available in the data. We will append to the matrices as
    # we loop through the events.
    context_matrices = collections.defaultdict(list)

    # List of Jaccard similarity scores for associations for all the events
    # under consideration
    jaccard_scores = []
    # Also track the average denominator when the numerator is 0 (to see how
    # many associations were even made for that event in the first place)
    jaccard_neg_denom = []
    # And when the denominator is positive
    jaccard_pos_denom = []

    # Loop over the paper set, finding (i) the list of available groundings;
    # and (ii) the set of events that were picked by all the human
    # annotators, added to the set of non-FP-marked Reach events

    # (Should match the total event count that we got when checking the
    # manual events as a group (by intersection) against Reach events); can
    # be printed for debugging
    total_event_count = 0

    for paper_id in paper_set:
        # Get list of groundings that were commonly available to all annotators
        # for this paper
        paper_groundings = get_available_groundings(data_path, annotators,
                                                    paper_id)

        # Get manual events that overlapped for all annotators (using the
        # same method as when getting the counts)
        def annotator_sort(annotator):
            return annotator_order[annotator]

        raw_manual_events = get_event_sets(data_path, annotators, paper_id)
        # Match manual_events lists to annotators
        manual_events = {}
        for x in range(len(raw_manual_events)):
            manual_events[annotators[x]] = raw_manual_events[x]

        raw_overlaps = get_event_overlaps(annotators, manual_events,
                                          annotator_order)
        intersect_key = tuple(sorted(annotators, key=annotator_sort))
        # Manual_event contains one row per annotator per event. We can loop
        # through this directly to check the associations.
        manual_events = raw_overlaps[intersect_key]

        # Add the Reach events
        reach_dict = get_reach_events(data_path, annotators, paper_id,
                                      by_annotator=True)
        # The Reach events are still in dictionary form key-ed to annotator ID.
        # Unroll them
        reach_events_list = reach_dict.values()

        paper_events = manual_events[:]
        for reach_event_list in reach_events_list:
            paper_events += reach_event_list

        total_event_count += len(manual_events)
        total_event_count += len(reach_events_list[0])

        # Now we can loop through the collection of events for this paper and
        # mark the category associations.
        # ---- Unfortunately, it's not clear how to check associations on
        # manual events, since they don't overlap exactly... Let's try it
        # with only Reach events first. ----
        for event_idx in range(len(reach_events_list[0])):

            # Sanity check: Let's make sure the intervals match up for all
            # the annotators on this event.
            test_match = reach_events_list[0][event_idx]

            for grounding in paper_groundings:

                # Generate the row list for this grounding/event pair
                this_row = [0, 0]

                for annotator_idx in range(len(annotators)):
                    this_annotator_event = \
                        reach_events_list[annotator_idx][event_idx]

                    # Sanity check: Repeated for each grounding, but that's OK
                    assert has_same_event_interval(
                        this_annotator_event,
                        test_match
                    )

                    if grounding in this_annotator_event.associations:
                        this_row[1] += 1
                    else:
                        this_row[0] += 1

                assert sum(this_row) == len(annotators)

                context_matrices[grounding].append(this_row)

            # At the same time, we can calculate the Jaccard similarity for
            # associations on this event if at least one association was made.
            this_association_sets = []

            for annotator_idx in range(len(annotators)):
                this_annotator_event = \
                    reach_events_list[annotator_idx][event_idx]

                # In particular, we only want groundings that were available
                # to all annotators.
                this_set = set(this_annotator_event.associations)
                this_set = this_set.intersection(paper_groundings)
                this_association_sets.append(this_set)

            intersection_sets = this_association_sets[:]
            intersection_set = intersection_sets.pop()
            for x in intersection_sets:
                intersection_set = intersection_set.intersection(x)

            union_sets = this_association_sets[:]
            union_set = union_sets.pop()
            for x in union_sets:
                union_set = union_set.union(x)

            if len(union_set) == 0:
                # None of the annotators made any associations for this event.
                # print("No associations. {}".format(this_association_sets))
                pass
            else:
                # print("Associations. {}".format(union_set))
                jaccard = len(intersection_set) / len(union_set)
                jaccard_scores.append(jaccard)

                if len(intersection_set) == 0:
                    # Record the denominator
                    jaccard_neg_denom.append(len(union_set))
                else:
                    jaccard_pos_denom.append(len(union_set))

    print("Computing Fleiss' Kappa...")
    context_kappas = {}
    context_association_counts = {}
    never_associated = []
    always_associated = []
    for grounding in context_matrices.keys():
        # We need to remove from consideration any groundings that were *never*
        # associated -- Or we get division by zero
        # Same for the off-chance that the grounding was *always* associated
        kappa_matrix = context_matrices[grounding][:]
        found_0 = False
        found_1 = False
        association_count = 0
        for row in kappa_matrix:
            # At the same time, keep track of association counts (by at least
            # one annotator)
            no_associations = [len(annotators), 0]
            if row != no_associations:
                association_count += 1

            # Don't break prematurely
            if row[0] != 0:
                found_0 = True
            if row[1] != 0:
                found_1 = True
            if found_0 and found_1:
                # break
                pass

        assert found_0 or found_1
        if found_0 and not found_1:
            never_associated.append(grounding)
        elif found_1 and not found_0:
            always_associated.append(grounding)
        else:
            context_kappas[grounding] = computeKappa(kappa_matrix)

        context_association_counts[grounding] = association_count

    # Sort and display kappa scores
    def kappa_sort(dict_item):
        # Given tuples of the form (<context>, <kappa>, ...)
        return dict_item[1]

    print("-----")
    print("[Context Kappa scores]")

    # [Display]: Include the association counts with each grounding ID entry,
    # and optionally display the associated free-text strings for each ID as
    # well
    display_scores = context_kappas.items()[:]
    display_temp = []
    for grounding, score in display_scores:
        # Add association by looking up raw ID
        association_count = context_association_counts[grounding]
        total_count = len(context_matrices[grounding])

        # Expand grounding to include free-text strings if requested
        if config.show_grounding_texts:
            grounding = "{} ({})".format(grounding, get_free_texts(grounding))

        display_temp.append((grounding, score, association_count, total_count))
    display_scores = display_temp

    # [Display]: Find longest grounding ID in the list
    id_width = 0
    for entry in display_scores:
        id_width = max(id_width, len(entry[0]))

    kappa_scores = []
    figure_data = []

    if not compact_score_list:
        for grounding, score, association_count, total_count in \
                sorted(display_scores, reverse=True, key=kappa_sort):
            print("{:<{}}: {:.03f} (Freq: {} out of {})"
                  "".format(grounding, id_width + 1, score, association_count,
                            total_count))
            kappa_scores.append(score)

            figure_data.append((float("{:.03f}".format(score)),
                                association_count))
    else:
        sorted_scores = sorted(display_scores, reverse=True,
                               key=kappa_sort)

        # Chop into two parts for compact printing
        lengths = divmod(len(sorted_scores), 2)
        cut_idx = lengths[0]
        if lengths[1] == 1:
            cut_idx += 1
        sorted_1 = sorted_scores[:cut_idx]
        sorted_2 = sorted_scores[cut_idx:]

        for print_idx in range(len(sorted_2)):
            grounding_1, score_1, count_1, total_1 = sorted_1[print_idx]
            grounding_2, score_2, count_2, total_2 = sorted_2[print_idx]
            print("{:<{}}: {:<{}.03f} {:<{}}: {:.03f} "
                  "".format(grounding_1, id_width + 1, score_1, 10,
                            grounding_2, id_width + 1, score_2))
            kappa_scores.append(score_1)
            kappa_scores.append(score_2)

            figure_data.append((float("{:.03f}".format(score_1)),
                                count_1))
            figure_data.append((float("{:.03f}".format(score_2)),
                                count_2))

        if lengths[1] == 1:
            # We have one more row in sorted_1
            grounding, score, count, total = sorted_1[-1]
            print("{:<{}}: {:.03f}".format(grounding, id_width + 1, score))
            kappa_scores.append(score)

            figure_data.append((float("{:.03f}".format(score)),
                                count))

    print()
    print("{:<{}}: {:.03f}".format("Average", id_width + 1,
                                   sum(kappa_scores) / len(kappa_scores)))

    # Visualisation
    import matplotlib.pyplot as plt
    import numpy as np

    # -- Histogram by frequency of Kappa score
    # Prepare the array
    min_count = 0
    num_bins = 15
    hist_data = []
    for score, count in figure_data:
        if count < min_count:
            continue
        for _ in range(count):
            hist_data.append(score)

    # http://stackoverflow.com/questions/6352740/matplotlib-label-each-bin
    fig, ax = plt.subplots(figsize=(10, 5))
    counts, bins, patches = plt.hist(hist_data, bins=num_bins,
                                     edgecolor='black',
                                     range=(-0.5, 1.0))

    # Set the ticks to be at the edges of the bins.
    # ax.set_xticks(bins)
    # Set the xaxis's tick labels to be formatted with 1 decimal place...
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%0.3f'))

    # Label the raw counts and the percentages below the x-axis...
    bin_centers = 0.5 * np.diff(bins) + bins[:-1]
    for count, x in zip(counts, bin_centers):
        # Label the raw counts
        if int(count) == 0:
            continue
        ax.annotate(str(int(count)), xy=(x, 0), xycoords=('data',
                                                          'axes fraction'),
                    xytext=(0, -24), textcoords='offset points', va='top',
                    ha='center')

        # # Label the percentages
        # percent = '%0.0f%%' % (100 * float(count) / counts.sum())
        # ax.annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),
        #             xytext=(0, -32), textcoords='offset points', va='top',
        #             ha='center')

    # Give ourselves some more room at the bottom of the plot
    plt.subplots_adjust(bottom=0.1, left=0.1, right=0.9, top=0.85)

    # Other adjustments
    plt.xticks(np.linspace(-0.5, 1, num=16))
    plt.xticks([-0.5, 0.0, 0.5, 1.0])
    plt.xlim(-0.5, 1)
    plt.axvline(x=0, linestyle='dashed', color='black', linewidth=0.5)
    title = \
        plt.title("Number of event associations for various context types\n"
                  "(binned by each context's Fleiss' Kappa score)")
    title.set_position([0.5, 1.05])
    title.set_size('x-large')
    title.set_weight('bold')

    # Save the plot
    plt.savefig("context_kappa_frequency.pdf")

    # -- Bar plot of groundings against Kappa, by Kappa descending
    plt.clf()
    plt.rc('text', usetex=True)

    min_count = 10

    raw_bar_data = [x for x in context_kappas.items()[:]
                    if context_association_counts[x[0]] > min_count]
    raw_bar_data = sorted(raw_bar_data, key=kappa_sort,
                          reverse=False)
    labels, kappas = zip(*raw_bar_data)

    # Make labels meaningful
    context_categories = [
        ("Species", ["taxonomy", "manual:Drosophila"]),
        ("Organ", ["uaz:UBERON", "uberon:"]),
        ("Tissue", ["tissuelist"]),
        ("Cell Type", ["uaz:CL"]),
        ("Cellular Component", ["go"]),
        ("Cell Line", ["atcc", "cellosaurus", "uaz:UA-CLine", "uaz:CVCL"]),
        ("Uniprot", ["uniprot:"])
    ]
    human_labels = []
    for label in labels:
        human_label = get_free_texts(label).split(",")[0]

        for category, prefixes in context_categories:
            for prefix in prefixes:
                if label.startswith(prefix):
                    human_label += " (\\textbf{{{}}})".format(category)

        human_labels.append(human_label)

    fig, ax = plt.subplots(figsize=(10, 4))
    plt.barh(range(len(labels)), kappas, height=0.8, tick_label=human_labels)

    plt.subplots_adjust(bottom=0.075, top=0.95, left=0.35, right=0.95)
    plt.ylim(-1, len(labels))
    plt.xlim(-0.5, 1)
    plt.xticks(np.linspace(-0.5, 1, num=16))

    locs, labels = plt.yticks()
    setter = [label.set_size('x-large') for label in labels]

    plt.savefig("bar_min10.pdf")

    exit()

    if show_edge_cases:
        print("-----")
        print("[Never associated]")
        if len(never_associated) == 0:
            print(" <None>")
        else:
            for grounding in sorted(never_associated):
                print(grounding)
        print("-----")
        print("[Always associated]")
        if len(always_associated) == 0:
            print(" <None>")
        else:
            for grounding in sorted(always_associated):
                print(grounding)

    print("-----")
    print("[Jaccard score]")
    print("Across events which had at least one association by at "
          "least one annotator (not necessarily shared):")
    print()
    print("  {:.03f}".format(sum(jaccard_scores) / len(jaccard_scores)))

    print()
    print("Average number of actual associations when no associations were "
          "shared amongst annotators:")
    print()
    print("  {:.03f}".format(sum(jaccard_neg_denom) / len(jaccard_neg_denom)))

    # ----
    jaccard_positive = [x for x in jaccard_scores
                        if x > 0]
    print()
    print("Across events which had at least one shared association amongst "
          "the annotators:")
    print()
    print("  {:.03f}".format(sum(jaccard_positive) / len(jaccard_positive)))

    print()
    print("Average number of actual associations when at least one "
          "association was shared amongst annotators:")
    print()
    print("  {:.03f}".format(sum(jaccard_pos_denom) / len(jaccard_pos_denom)))

    # Debug: Interactive mode to see the actual context matrix for a given
    # grounding.
    # pprint.pprint(context_matrices['cellosaurus:CVCL_1220'])
    print("-----")
    print("Interactive mode -- Specify a grounding ID to see its Kappa matrix")
    print("('quit' or 'exit' to exit)")
    quit_flag = False
    while not quit_flag:
        show_matrix = raw_input("> ")
        if show_matrix in context_matrices.keys():
            pprint.pprint(context_matrices[show_matrix])
        elif show_matrix == "quit" or show_matrix == "exit":
            quit_flag = True
        else:
            print("Invalid grounding ID.")


def has_same_event_interval(event_1, event_2):
    """
    Given 2 event Namespaces, checks to ensure that their line_num, start,
    and end are the same. Other attributes are ignored.
    :param event_1:
    :param event_2:
    :return:
    """
    return (event_1.line_num == event_2.line_num and
            event_1.start == event_2.start and
            event_1.end == event_2.end)


if __name__ == "__main__":
    main()
