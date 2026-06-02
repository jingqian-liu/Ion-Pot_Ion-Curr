from ion_pot.model import MultiScaleConv3dEPBModelConfig

epb_model = MultiScaleConv3dEPBModelConfig(
    hidden_dim=16,
    atom_type_num=8,
    reaction_size=9,
    patch_size=128,
    lvlset_embed_scale=10,
    lvlset_embed_theta=10000,
    charge_embed_scale=1000,
    charge_embed_theta=10000,
    kernel_sizes=[1, 5, 7],
    embed_dim_scale=2,
)

epb_psz128_model = MultiScaleConv3dEPBModelConfig(
    hidden_dim=16,
    atom_type_num=8,
    reaction_size=9,
    patch_size=128,
    lvlset_embed_scale=10,
    lvlset_embed_theta=10000,
    charge_embed_scale=1000,
    charge_embed_theta=10000,
    kernel_sizes=[1, 5, 7],
    embed_dim_scale=2,
    dropout_rate=0.1,
    block_drop_rate=0.1,
)

epb_psz128_model_medium = MultiScaleConv3dEPBModelConfig(
    hidden_dim=32,
    atom_type_num=8,
    reaction_size=9,
    patch_size=128,
    lvlset_embed_scale=10,
    lvlset_embed_theta=10000,
    charge_embed_scale=1000,
    charge_embed_theta=10000,
    kernel_sizes=[1, 5, 7],
    embed_dim_scale=4,
    dropout_rate=0.1,
    block_drop_rate=0.1,
)

epb_psz128_all_in_one_model_medium = MultiScaleConv3dEPBModelConfig(
    hidden_dim=32,
    atom_type_num=153,
    reaction_size=9,
    patch_size=128,
    lvlset_embed_scale=10,
    lvlset_embed_theta=10000,
    charge_embed_scale=1000,
    charge_embed_theta=10000,
    kernel_sizes=[1, 5, 7],
    embed_dim_scale=4,
    dropout_rate=0.1,
    block_drop_rate=0.1,
)

epb_psz128_all_in_one_wo_lset_model_medium = MultiScaleConv3dEPBModelConfig(
    hidden_dim=32,
    atom_type_num=153,
    reaction_size=9,
    patch_size=128,
    lvlset_embed_scale=10,
    lvlset_embed_theta=10000,
    charge_embed_scale=1000,
    charge_embed_theta=10000,
    kernel_sizes=[1, 5, 7],
    embed_dim_scale=4,
    dropout_rate=0.1,
    block_drop_rate=0.1,
    use_lset=False,
)

epb_psz64_all_in_one_wo_lset_model_medium = MultiScaleConv3dEPBModelConfig(
    hidden_dim=32,
    atom_type_num=153,
    reaction_size=9,
    patch_size=64,
    lvlset_embed_scale=10,
    lvlset_embed_theta=10000,
    charge_embed_scale=1000,
    charge_embed_theta=10000,
    kernel_sizes=[1, 5, 7],
    embed_dim_scale=4,
    dropout_rate=0.1,
    block_drop_rate=0.1,
    use_lset=False,
)

epb_psz64_small_mol_model_medium = MultiScaleConv3dEPBModelConfig(
    hidden_dim=32,
    atom_type_num=153,
    reaction_size=3,
    patch_size=64,
    lvlset_embed_scale=10,
    lvlset_embed_theta=10000,
    charge_embed_scale=1000,
    charge_embed_theta=10000,
    kernel_sizes=[1, 1, 3],
    use_avg_pool=True,
    pooling_scale=2,
    embed_dim_scale=4,
    dropout_rate=0.1,
    block_drop_rate=0.1,
)

epb_psz64_small_mol_wo_lset_model_medium = MultiScaleConv3dEPBModelConfig(
    hidden_dim=32,
    atom_type_num=153,
    reaction_size=3,
    patch_size=64,
    lvlset_embed_scale=10,
    lvlset_embed_theta=10000,
    charge_embed_scale=1000,
    charge_embed_theta=10000,
    kernel_sizes=[1, 1, 3],
    use_avg_pool=True,
    pooling_scale=2,
    embed_dim_scale=4,
    dropout_rate=0.1,
    block_drop_rate=0.1,
    use_lset=False,
)

epb_psz64_model = MultiScaleConv3dEPBModelConfig(
    hidden_dim=16,
    atom_type_num=8,
    reaction_size=9,
    patch_size=64,
    lvlset_embed_scale=10,
    lvlset_embed_theta=10000,
    charge_embed_scale=1000,
    charge_embed_theta=10000,
    kernel_sizes=[1, 5, 7],
    embed_dim_scale=2,
    dropout_rate=0.1,
)

epb_psz64_model_medium = MultiScaleConv3dEPBModelConfig(
    hidden_dim=32,
    atom_type_num=8,
    reaction_size=9,
    patch_size=64,
    lvlset_embed_scale=10,
    lvlset_embed_theta=10000,
    charge_embed_scale=1000,
    charge_embed_theta=10000,
    kernel_sizes=[1, 5, 7],
    embed_dim_scale=4,
    dropout_rate=0.1,
)
