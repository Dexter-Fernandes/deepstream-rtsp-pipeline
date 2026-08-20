import numpy as np

from pipelines.mtmc import (
    GroundObservation,
    MtmcConfig,
    MtmcFuser,
    _candidate_cost,
    fuse_offline,
)


def test_separated_people_keep_distinct_global_ids_across_cameras():
    observations = []
    for frame in range(5):
        timestamp_ns = frame * 100_000_000
        observations.extend(
            [
                GroundObservation(timestamp_ns, 0, 10, frame * 0.1, 0.0, 0.1),
                GroundObservation(timestamp_ns, 1, 20, frame * 0.1 + 0.02, 0.0, 0.1),
                GroundObservation(timestamp_ns, 0, 11, frame * 0.1, 5.0, 0.1),
                GroundObservation(timestamp_ns, 1, 21, frame * 0.1 + 0.02, 5.0, 0.1),
            ]
        )

    id_map = fuse_offline(observations, MtmcConfig())

    assert id_map[(0, 10)] == id_map[(1, 20)]
    assert id_map[(0, 11)] == id_map[(1, 21)]
    assert id_map[(0, 10)] != id_map[(0, 11)]


def test_crossing_people_do_not_collapse_into_one_identity():
    observations = []
    source_1_positions = [(0.0, 4.0), (1.0, 3.0), (2.2, 1.8), (3.2, 0.8), (4.0, 0.0)]
    for frame, (source_1_a, source_1_b) in enumerate(source_1_positions):
        timestamp_ns = frame * 100_000_000
        source_0_a = float(frame)
        source_0_b = float(4 - frame)
        observations.extend(
            [
                GroundObservation(timestamp_ns, 0, 10, source_0_a, 0.0, 0.5),
                GroundObservation(timestamp_ns, 0, 11, source_0_b, 0.0, 0.5),
                GroundObservation(timestamp_ns, 1, 20, source_1_a, 0.0, 0.5),
                GroundObservation(timestamp_ns, 1, 21, source_1_b, 0.0, 0.5),
            ]
        )

    id_map = fuse_offline(
        observations,
        MtmcConfig(min_support=1, min_affinity=0.2, max_distance_m=3.0),
    )

    assert len(set(id_map.values())) == 2
    assert id_map[(0, 10)] != id_map[(0, 11)]
    assert id_map[(1, 20)] != id_map[(1, 21)]


def test_observations_above_maximum_sigma_are_dropped():
    observations = [
        GroundObservation(frame * 100_000_000, 0, 10, 0.0, 0.0, 0.1)
        for frame in range(5)
    ]
    observations.extend(
        GroundObservation(frame * 100_000_000, 1, 20, 0.0, 0.0, 1.51)
        for frame in range(5)
    )

    id_map = fuse_offline(observations, MtmcConfig(max_sigma_m=1.5))

    assert (0, 10) in id_map
    assert (1, 20) not in id_map


def test_rejected_observation_does_not_leak_an_embedding_cache_entry():
    fuser = MtmcFuser(MtmcConfig(max_sigma_m=1.5))

    fuser.observe(
        0,
        0,
        [GroundObservation(0, 0, 10, 0.0, 0.0, 1.51, np.array([1.0, 0.0]))],
    )

    assert fuser._embedding_cache == {}


def test_cosine_appearance_cost_disambiguates_equal_geometry():
    observations = []
    for frame in range(2):
        timestamp_ns = frame * 100_000_000
        observations.extend(
            [
                GroundObservation(timestamp_ns, 0, 10, 0.0, 0.0, 0.1, np.array([1.0, 0.0])),
                GroundObservation(timestamp_ns, 0, 11, 0.0, 0.0, 0.1, np.array([0.0, 1.0])),
                GroundObservation(timestamp_ns, 1, 20, 0.0, 0.0, 0.1, np.array([0.0, 1.0])),
                GroundObservation(timestamp_ns, 1, 21, 0.0, 0.0, 0.1, np.array([1.0, 0.0])),
            ]
        )

    id_map = fuse_offline(
        observations,
        MtmcConfig(
            w_geo=1.0,
            w_app=1.0,
            min_cooccurrence=2,
            min_support=2,
            min_affinity=1.0,
        ),
    )

    assert id_map[(0, 10)] == id_map[(1, 21)]
    assert id_map[(0, 11)] == id_map[(1, 20)]


def test_missing_appearance_is_neutral_instead_of_a_perfect_match():
    config = MtmcConfig(w_app=0.5)
    anchor = GroundObservation(0, 0, 10, 0.0, 0.0, 0.1, np.array([1.0, 0.0]))
    matching = GroundObservation(0, 1, 20, 0.0, 0.0, 0.1, np.array([1.0, 0.0]))
    missing = GroundObservation(0, 1, 21, 0.0, 0.0, 0.1)
    opposite = GroundObservation(0, 1, 22, 0.0, 0.0, 0.1, np.array([-1.0, 0.0]))

    matching_cost = _candidate_cost(anchor, matching, config)
    missing_cost = _candidate_cost(anchor, missing, config)
    opposite_cost = _candidate_cost(anchor, opposite, config)

    assert matching_cost < missing_cost < opposite_cost


def test_online_fuser_carries_sparse_embeddings_forward_per_tracklet():
    fuser = MtmcFuser(
        MtmcConfig(
            sync_bucket_ns=100,
            w_app=1.0,
            min_cooccurrence=5,
            min_support=5,
            min_affinity=1.0,
            reassign_interval=5,
        ),
        expected_sources={0, 1},
    )
    for frame in range(5):
        timestamp_ns = frame * 100
        embedding_a = np.array([1.0, 0.0]) if frame == 0 else None
        embedding_b = np.array([0.0, 1.0]) if frame == 0 else None
        fuser.observe(
            timestamp_ns,
            0,
            [
                GroundObservation(timestamp_ns, 0, 10, 0.0, 0.0, 0.1, embedding_a),
                GroundObservation(timestamp_ns, 0, 11, 0.0, 0.0, 0.1, embedding_b),
            ],
        )
        fuser.observe(
            timestamp_ns,
            1,
            [
                GroundObservation(timestamp_ns, 1, 20, 0.0, 0.0, 0.1, embedding_b),
                GroundObservation(timestamp_ns, 1, 21, 0.0, 0.0, 0.1, embedding_a),
            ],
        )
        fuser.flush_ready()

    assert fuser.global_id(0, 10) == fuser.global_id(1, 21)
    assert fuser.global_id(0, 11) == fuser.global_id(1, 20)


def test_online_fuser_publishes_a_complete_bucket_assignment():
    fuser = MtmcFuser(
        MtmcConfig(
            min_cooccurrence=1,
            min_support=1,
            min_affinity=1.0,
            reassign_interval=1,
        ),
        expected_sources={0, 1},
    )
    fuser.observe(0, 0, [GroundObservation(0, 0, 10, 1.0, 2.0, 0.1)])
    fuser.observe(0, 1, [GroundObservation(0, 1, 20, 1.0, 2.0, 0.1)])

    assert fuser.flush_ready() == [0]
    assert fuser.global_id(0, 10) == fuser.global_id(1, 20)


def test_incomplete_bucket_flushes_after_maximum_skew():
    fuser = MtmcFuser(
        MtmcConfig(sync_bucket_ns=100, max_skew_ns=150),
        expected_sources={0, 1},
    )
    fuser.observe(0, 0, [GroundObservation(0, 0, 10, 0.0, 0.0, 0.1)])
    fuser.observe(201, 0, [])

    assert fuser.flush_ready() == [0]
    assert fuser.stats()["desync_buckets"] == 1


def test_complete_bucket_refuses_fusion_when_source_timestamps_exceed_skew():
    fuser = MtmcFuser(
        MtmcConfig(
            sync_bucket_ns=1_000,
            max_skew_ns=100,
            min_cooccurrence=1,
            min_support=1,
            min_affinity=1.0,
            reassign_interval=1,
        ),
        expected_sources={0, 1},
    )
    fuser.observe(0, 0, [GroundObservation(0, 0, 10, 0.0, 0.0, 0.1)])
    fuser.observe(200, 1, [GroundObservation(200, 1, 20, 0.0, 0.0, 0.1)])

    assert fuser.flush_ready() == [0]
    assert fuser.global_id(0, 10) != fuser.global_id(1, 20)
    assert fuser.stats()["skew_refusals"] == 1


def test_reconnect_generation_prevents_recycled_object_id_from_using_stale_identity():
    fuser = MtmcFuser(
        MtmcConfig(
            min_cooccurrence=1,
            min_support=1,
            min_affinity=1.0,
            reassign_interval=1,
        ),
        expected_sources={0, 1},
    )
    fuser.observe(0, 0, [GroundObservation(0, 0, 7, 0.0, 0.0, 0.1)])
    fuser.observe(0, 1, [GroundObservation(0, 1, 8, 0.0, 0.0, 0.1)])
    fuser.flush_ready()
    old_global_id = fuser.global_id(0, 7)

    assert fuser.bump_generation(0) == 1
    assert fuser.global_id(0, 7) is None
    assert (0, 7) not in fuser.id_map()
    assert fuser.qualified_id_map()[(0, 0, 7)] == old_global_id


def test_recycled_object_id_after_reconnect_is_a_distinct_tracklet():
    config = MtmcConfig(
        min_cooccurrence=1,
        min_support=1,
        min_affinity=1.0,
        reassign_interval=1,
    )
    fuser = MtmcFuser(config, expected_sources={0, 1})
    fuser.observe(0, 0, [GroundObservation(0, 0, 7, 0.0, 0.0, 0.1)])
    fuser.observe(0, 1, [GroundObservation(0, 1, 8, 0.0, 0.0, 0.1)])
    fuser.flush_ready()
    old_global_id = fuser.global_id(0, 7)

    fuser.bump_generation(0)
    fuser.observe(1_000_000_000, 0, [GroundObservation(0, 0, 7, 10.0, 0.0, 0.1)])
    fuser.observe(1_000_000_000, 1, [GroundObservation(0, 1, 9, 10.0, 0.0, 0.1)])
    fuser.flush_ready()

    qualified = fuser.qualified_id_map()
    assert qualified[(0, 0, 7)] == old_global_id
    assert qualified[(0, 1, 7)] != old_global_id
    assert fuser.global_id(0, 7) == qualified[(0, 1, 7)]


def test_tracklet_ttl_evicts_association_state_and_published_entries():
    fuser = MtmcFuser(
        MtmcConfig(
            reassign_interval=1,
            tracklet_ttl=2,
            sync_bucket_ns=100,
        ),
        expected_sources={0},
    )
    fuser.observe(0, 0, [GroundObservation(0, 0, 7, 0.0, 0.0, 0.1)])
    fuser.flush_ready()
    assert fuser.global_id(0, 7) is not None

    fuser.observe(100, 0, [])
    fuser.flush_ready()
    assert fuser.global_id(0, 7) is not None

    fuser.observe(200, 0, [])
    fuser.flush_ready()
    assert fuser.global_id(0, 7) is None
    assert (0, 0, 7) not in fuser.qualified_id_map()
    assert fuser.stats()["active_tracklets"] == 0


def test_periodic_reassignment_keeps_existing_global_id_sticky():
    fuser = MtmcFuser(
        MtmcConfig(
            min_cooccurrence=1,
            min_support=1,
            min_affinity=0.5,
            reassign_interval=1,
        ),
        expected_sources={0, 1, 2},
    )
    fuser.observe(0, 0, [])
    fuser.observe(0, 1, [GroundObservation(0, 1, 20, 0.0, 0.0, 0.1)])
    fuser.observe(0, 2, [GroundObservation(0, 2, 30, 0.0, 0.0, 0.1)])
    fuser.flush_ready()
    original_global_id = fuser.global_id(1, 20)

    timestamp_ns = 33_000_000
    fuser.observe(timestamp_ns, 0, [GroundObservation(0, 0, 1, 10.0, 0.0, 0.1)])
    fuser.observe(timestamp_ns, 1, [GroundObservation(0, 1, 20, 0.0, 0.0, 0.1)])
    fuser.observe(timestamp_ns, 2, [GroundObservation(0, 2, 30, 0.0, 0.0, 0.1)])
    fuser.flush_ready()

    assert fuser.global_id(1, 20) == original_global_id
    assert fuser.global_id(2, 30) == original_global_id
    assert fuser.global_id(0, 1) != original_global_id


def test_merged_cluster_inherits_the_majority_prior_global_id():
    config = MtmcConfig(
        sync_bucket_ns=100,
        min_cooccurrence=1,
        min_support=1,
        min_affinity=0.5,
        reassign_interval=1,
    )
    fuser = MtmcFuser(config, expected_sources={0, 1, 2})
    fuser.observe(0, 0, [GroundObservation(0, 0, 10, 0.0, 0.0, 0.1)])
    fuser.observe(0, 1, [GroundObservation(0, 1, 20, 0.0, 0.0, 0.1)])
    fuser.observe(0, 2, [GroundObservation(0, 2, 30, 10.0, 0.0, 0.1)])
    fuser.flush_ready()
    majority_id = fuser.global_id(0, 10)
    minority_id = fuser.global_id(2, 30)
    assert majority_id != minority_id

    fuser.observe(100, 0, [GroundObservation(100, 0, 10, 0.0, 0.0, 0.1)])
    fuser.observe(100, 1, [GroundObservation(100, 1, 20, 0.0, 0.0, 0.1)])
    fuser.observe(100, 2, [GroundObservation(100, 2, 30, 0.0, 0.0, 0.1)])
    fuser.flush_ready()

    assert fuser.global_id(0, 10) == majority_id
    assert fuser.global_id(1, 20) == majority_id
    assert fuser.global_id(2, 30) == majority_id


def test_equal_stickiness_votes_choose_the_oldest_global_id():
    config = MtmcConfig(
        sync_bucket_ns=100,
        min_cooccurrence=1,
        min_support=1,
        min_affinity=0.5,
        reassign_interval=1,
    )
    fuser = MtmcFuser(config, expected_sources={0, 1})
    fuser.observe(0, 0, [GroundObservation(0, 0, 10, 0.0, 0.0, 0.1)])
    fuser.observe(0, 1, [GroundObservation(0, 1, 20, 10.0, 0.0, 0.1)])
    fuser.flush_ready()
    prior_ids = {fuser.global_id(0, 10), fuser.global_id(1, 20)}

    fuser.observe(100, 0, [GroundObservation(100, 0, 10, 0.0, 0.0, 0.1)])
    fuser.observe(100, 1, [GroundObservation(100, 1, 20, 0.0, 0.0, 0.1)])
    fuser.flush_ready()

    assert fuser.global_id(0, 10) == min(prior_ids)
    assert fuser.global_id(1, 20) == min(prior_ids)


def test_offline_fusion_refuses_a_bucket_above_maximum_timestamp_skew():
    observations = [
        GroundObservation(0, 0, 10, 0.0, 0.0, 0.1),
        GroundObservation(200, 1, 20, 0.0, 0.0, 0.1),
    ]
    config = MtmcConfig(
        sync_bucket_ns=1_000,
        max_skew_ns=100,
        min_cooccurrence=1,
        min_support=1,
        min_affinity=1.0,
    )

    id_map = fuse_offline(observations, config)

    assert id_map[(0, 10)] != id_map[(1, 20)]


def test_skew_guard_uses_all_observation_timestamps_in_a_bucket():
    fuser = MtmcFuser(
        MtmcConfig(
            sync_bucket_ns=1_000,
            max_skew_ns=100,
            min_cooccurrence=1,
            min_support=1,
            min_affinity=1.0,
            reassign_interval=1,
        ),
        expected_sources={0, 1},
    )
    fuser.observe(0, 0, [GroundObservation(0, 0, 10, 0.0, 0.0, 0.1)])
    fuser.observe(90, 0, [GroundObservation(90, 0, 10, 0.0, 0.0, 0.1)])
    fuser.observe(150, 1, [GroundObservation(150, 1, 20, 0.0, 0.0, 0.1)])

    fuser.flush_ready()

    assert fuser.global_id(0, 10) != fuser.global_id(1, 20)
    assert fuser.stats()["skew_refusals"] == 1


def test_force_flush_publishes_final_assignment_before_reassign_interval():
    fuser = MtmcFuser(
        MtmcConfig(reassign_interval=10),
        expected_sources={0},
    )
    fuser.observe(0, 0, [GroundObservation(0, 0, 7, 0.0, 0.0, 0.1)])
    assert fuser.flush_ready() == [0]
    assert fuser.global_id(0, 7) is None

    assert fuser.flush_ready(force=True) == []
    assert fuser.global_id(0, 7) is not None


def test_overlapping_same_camera_tracklets_never_merge_through_a_bridge():
    observations = [
        GroundObservation(0, 0, 10, 0.0, 0.0, 0.1),
        GroundObservation(0, 1, 20, 0.0, 0.0, 0.1),
        GroundObservation(100, 0, 11, 0.0, 0.0, 0.1),
        GroundObservation(100, 1, 20, 0.0, 0.0, 0.1),
        GroundObservation(200, 0, 10, 0.0, 0.0, 0.1),
        GroundObservation(200, 1, 20, 0.0, 0.0, 0.1),
        GroundObservation(300, 0, 11, 0.0, 0.0, 0.1),
        GroundObservation(300, 1, 20, 0.0, 0.0, 0.1),
    ]
    config = MtmcConfig(
        sync_bucket_ns=100,
        min_cooccurrence=2,
        min_support=2,
        min_affinity=1.0,
    )

    id_map = fuse_offline(observations, config)

    assert id_map[(0, 10)] != id_map[(0, 11)]


def test_sequential_same_camera_fragments_can_merge_through_another_camera():
    observations = []
    for frame, object_id in enumerate((10, 10, 11, 11)):
        timestamp_ns = frame * 100
        observations.extend(
            [
                GroundObservation(timestamp_ns, 0, object_id, float(frame), 0.0, 0.1),
                GroundObservation(timestamp_ns, 1, 20, float(frame), 0.0, 0.1),
            ]
        )
    config = MtmcConfig(
        sync_bucket_ns=100,
        min_cooccurrence=2,
        min_support=2,
        min_affinity=1.0,
    )

    id_map = fuse_offline(observations, config)

    assert id_map[(0, 10)] == id_map[(0, 11)] == id_map[(1, 20)]


def test_support_never_merges_before_minimum_cooccurrence():
    observations = []
    for frame in range(4):
        timestamp_ns = frame * 100
        observations.extend(
            [
                GroundObservation(timestamp_ns, 0, 10, 0.0, 0.0, 0.1),
                GroundObservation(timestamp_ns, 1, 20, 0.0, 0.0, 0.1),
            ]
        )

    id_map = fuse_offline(
        observations,
        MtmcConfig(
            sync_bucket_ns=100,
            min_cooccurrence=5,
            min_support=1,
            min_affinity=0.0,
        ),
    )

    assert id_map[(0, 10)] != id_map[(1, 20)]


def test_affinity_is_not_defined_before_minimum_support():
    observations = [
        GroundObservation(0, 0, 10, 0.0, 0.0, 0.1),
        GroundObservation(0, 1, 20, 0.0, 0.0, 0.1),
        GroundObservation(100, 0, 10, 0.0, 0.0, 0.1),
        GroundObservation(100, 1, 20, 10.0, 0.0, 0.1),
    ]

    id_map = fuse_offline(
        observations,
        MtmcConfig(
            sync_bucket_ns=100,
            min_cooccurrence=2,
            min_support=2,
            min_affinity=0.0,
        ),
    )

    assert id_map[(0, 10)] != id_map[(1, 20)]


def test_absolute_distance_cap_rejects_far_field_match_despite_large_uncertainty():
    observations = [
        GroundObservation(0, 0, 10, 0.0, 0.0, 1.0),
        GroundObservation(0, 1, 20, 1.5, 0.0, 0.1),
    ]
    config = MtmcConfig(
        z_gate=3.0,
        max_distance_m=1.0,
        min_cooccurrence=1,
        min_support=1,
        min_affinity=1.0,
    )

    id_map = fuse_offline(observations, config)

    assert id_map[(0, 10)] != id_map[(1, 20)]


def test_geometry_gate_is_normalised_by_combined_uncertainty():
    observations = [
        GroundObservation(0, 0, 10, 0.0, 0.0, 0.2),
        GroundObservation(0, 1, 20, 0.5, 0.0, 0.2),
        GroundObservation(0, 0, 11, 0.0, 5.0, 0.1),
        GroundObservation(0, 1, 21, 0.5, 5.0, 0.1),
    ]
    config = MtmcConfig(
        z_gate=2.0,
        min_cooccurrence=1,
        min_support=1,
        min_affinity=1.0,
    )

    id_map = fuse_offline(observations, config)

    assert id_map[(0, 10)] == id_map[(1, 20)]
    assert id_map[(0, 11)] != id_map[(1, 21)]


def test_offline_fusion_is_deterministic_for_reversed_input_order():
    observations = []
    for frame in range(5):
        timestamp_ns = frame * 100
        observations.extend(
            [
                GroundObservation(timestamp_ns, 2, 30, 4.0, 0.0, 0.1),
                GroundObservation(timestamp_ns, 0, 10, 0.0, 0.0, 0.1),
                GroundObservation(timestamp_ns, 1, 20, 0.0, 0.0, 0.1),
                GroundObservation(timestamp_ns, 1, 21, 4.0, 0.0, 0.1),
            ]
        )
    config = MtmcConfig(sync_bucket_ns=100)

    forward = fuse_offline(observations, config)
    reverse = fuse_offline(reversed(observations), config)

    assert forward == reverse


def test_id_map_returns_isolated_immutable_snapshots_across_publication():
    fuser = MtmcFuser(MtmcConfig(reassign_interval=1), expected_sources={0})
    fuser.observe(0, 0, [GroundObservation(0, 0, 7, 0.0, 0.0, 0.1)])
    fuser.flush_ready()

    first = fuser.id_map()
    second = fuser.id_map()
    assert first == second
    assert first is not second

    fuser.bump_generation(0)
    assert first
    assert fuser.id_map() == {}
    first.clear()
    assert fuser.qualified_id_map()


def test_out_of_order_per_source_offset_timestamps_still_share_buckets():
    observations = [
        GroundObservation(3_050, 1, 20, 2.0, 0.0, 0.1),
        GroundObservation(1_010, 0, 10, 0.0, 0.0, 0.1),
        GroundObservation(2_050, 1, 20, 1.0, 0.0, 0.1),
        GroundObservation(3_010, 0, 10, 2.0, 0.0, 0.1),
        GroundObservation(1_050, 1, 20, 0.0, 0.0, 0.1),
        GroundObservation(2_010, 0, 10, 1.0, 0.0, 0.1),
    ]
    config = MtmcConfig(
        sync_bucket_ns=1_000,
        max_skew_ns=100,
        min_cooccurrence=3,
        min_support=3,
        min_affinity=1.0,
    )

    id_map = fuse_offline(observations, config)

    assert id_map[(0, 10)] == id_map[(1, 20)]


def test_late_older_bucket_does_not_rewind_tracklet_ttl_watermark():
    fuser = MtmcFuser(
        MtmcConfig(
            sync_bucket_ns=100,
            max_skew_ns=1_000,
            min_cooccurrence=1,
            min_support=1,
            min_affinity=1.0,
            reassign_interval=1,
            tracklet_ttl=2,
        ),
        expected_sources={0, 1},
    )
    fuser.observe(200, 0, [GroundObservation(200, 0, 10, 0.0, 0.0, 0.1)])
    fuser.observe(200, 1, [GroundObservation(200, 1, 20, 0.0, 0.0, 0.1)])
    fuser.flush_ready()
    current_global_id = fuser.global_id(0, 10)

    fuser.observe(0, 0, [GroundObservation(0, 0, 10, 0.0, 0.0, 0.1)])
    fuser.observe(0, 1, [GroundObservation(0, 1, 20, 0.0, 0.0, 0.1)])
    fuser.flush_ready()

    assert fuser.global_id(0, 10) == current_global_id
    assert fuser.global_id(1, 20) == current_global_id


def test_mutual_nearest_matching_declines_a_greedy_second_choice():
    observations = [
        GroundObservation(0, 0, 10, 0.0, 0.0, 1.0),
        GroundObservation(0, 0, 11, 2.0, 0.0, 1.0),
        GroundObservation(0, 1, 20, 0.9, 0.0, 1.0),
        GroundObservation(0, 1, 21, 10.0, 0.0, 1.0),
    ]
    config = MtmcConfig(
        z_gate=20.0,
        max_distance_m=20.0,
        max_sigma_m=2.0,
        min_cooccurrence=1,
        min_support=1,
        min_affinity=1.0,
    )

    id_map = fuse_offline(observations, config)

    assert id_map[(0, 10)] == id_map[(1, 20)]
    assert id_map[(0, 11)] != id_map[(1, 21)]
