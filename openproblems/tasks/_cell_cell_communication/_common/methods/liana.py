from .....tools.conversion import r_function
from .....tools.decorators import method
from .....tools.normalize import log_cp10k
from .....tools.utils import check_r_version
from ..utils import aggregate_method_scores
from ..utils import ligand_receptor_resource

import functools

import liana as li
from liana.multi._common import _process_scores
from liana.method import rank_aggregate, singlecellsignalr, connectome, cellphonedb, natmi, logfc, cellchat, geometric_mean

# Helper function to filter according to permutation p-values
def _p_filt(x, y):
    if x <= 0.05:
        return y
    else:
        return 0

_liana_method = functools.partial(
    method,
    method_summary=(
        "RobustRankAggregate generates a consensus rank of all methods implemented in"
        " LIANA providing either specificity or magnitude scores."
    ),
    paper_name=(
        "Comparison of methods and resources for cell-cell communication inference from"
        " single-cell RNA-Seq data"
    ),
    paper_reference="dimitrov2022comparison",
    paper_year=2022,
    code_url="https://github.com/saezlab/liana-py",
    image="openproblems-r-extras", # TODO discuss image with Scott
)


def _liana(
    adata,
    method,
    score_key,
    test=False,
    min_expression_prop=0.1,
):
    # log-normalize
    adata = log_cp10k(adata)
    
    if test:
        n_perms = 2
    else:
        n_perms = 1000

    # TODO replace this with pypath
    resource = ligand_receptor_resource(adata.uns["target_organism"])
    resource = (resource[["source_genesymbol", "target_genesymbol"]].
                rename(columns={"source_genesymbol": "ligand", "target_genesymbol": "receptor"}))

    # Run LIANA
    method(adata=adata,
           groupby="label",
           resource=resource,
           expr_prop=min_expression_prop,
           key_added='ccc_pred',
           n_perms=n_perms,
           layer='log_cp10k',
           use_raw=False,
           )
    liana_res = adata.uns['ccc_pred']
    
    # deal with any scores in ascending order (typically probabilities)
    adata.uns['ccc_pred'] = _process_scores(liana_res,
                                            score_key=score_key,
                                            inverse_fun=lambda x: 1 - x
                                            )
    
    # apply p-value filter to those that use it
    pval_methods = [cellphonedb.magnitude,
                    # TODO discuss: it's fair but magnitude rank will represent be magnitude alone
                    rank_aggregate.magnitude,
                    geometric_mean.magnitude,
                    cellchat.magnitude]
    if score_key in pval_methods:
        adata.uns["ccc_pred"][method.magnitude] = adata.uns["ccc_pred"].apply(
        lambda x: _p_filt(x[method.specificity], x[method.magnitude]), axis=1)
    
    liana_res[['ligand', 'receptor']] = \
        liana_res[['ligand_complex', 'receptor_complex']]
    liana_res = liana_res.rename(columns={score_key: 'score'})
    
    adata.uns["ccc_pred"] = liana_res
    adata.uns["method_code_version"] = li.__version__

    return adata


@_liana_method(
    method_name="Specificity Rank Aggregate (max)",
)
def specificity_max(adata, test=False):
    adata = _liana(adata=adata, 
                   method=rank_aggregate,
                   test=test,
                   score_key=rank_aggregate.specificity
                   )
    adata.uns["ccc_pred"] = aggregate_method_scores(adata, how="max")

    return adata


@_liana_method(
    method_name="Specificity Rank Aggregate (sum)",
)
def specificity_sum(adata, test=False):
    adata = _liana(adata=adata, 
                   method=rank_aggregate,
                   test=test,
                   score_key=rank_aggregate.specificity
                   )
    adata.uns["ccc_pred"] = aggregate_method_scores(adata, how="sum")

    return adata


@_liana_method(
    method_name="Magnitude Rank Aggregate (max)",
)
def magnitude_max(adata, test=False):
    adata = _liana(adata=adata,
                   method=rank_aggregate,
                   test=test,
                   score_key=rank_aggregate.magnitude
                   )
    adata.uns["ccc_pred"] = aggregate_method_scores(adata, how="max")

    return adata


@_liana_method(
    method_name="Magnitude Rank Aggregate (sum)",
)
def magnitude_sum(adata, test=False):
    adata = _liana(adata=adata,
                    method=rank_aggregate,
                    test=test,
                    score_key=rank_aggregate.magnitude
                    )
    adata.uns["ccc_pred"] = aggregate_method_scores(adata, how="sum")

    return adata


_cellphonedb_method = functools.partial(
    _liana_method,
    method_summary=(
        "CellPhoneDBv2 calculates a mean of ligand-receptor expression as a measure of"
        " interaction magnitude, along with a permutation-based p-value as a measure of"
        " specificity. Here, we use the former to prioritize interactions, subsequent"
        " to filtering according to p-value less than 0.05."
    ),
    paper_name=(
        "CellPhoneDB: inferring cell–cell communication from combined expression of"
        " multi-subunit ligand–receptor complexes"
    ),
    paper_reference="efremova2020cellphonedb",
    paper_year=2020,
)


def _cellphonedb(adata, test=False):
    adata = _liana(adata=adata,
                   method=cellphonedb,
                   test=test,
                   score_key=cellphonedb.magnitude
                   )
    return adata


@_cellphonedb_method(
    method_name="CellPhoneDB (max)",
)
def cellphonedb_max(adata, test=False):
    adata = _cellphonedb(adata, test=test)
    adata.uns["ccc_pred"] = aggregate_method_scores(adata, how="max")

    return adata


@_cellphonedb_method(
    method_name="CellPhoneDB (sum)",
)
def cellphonedb_sum(adata, test=False):
    adata = _cellphonedb(adata, test=test)
    adata.uns["ccc_pred"] = aggregate_method_scores(adata, how="sum")

    return adata


_connectome_method = functools.partial(
    _liana_method,
    method_summary=(
        "Connectome uses the product of ligand-receptor expression as a measure of"
        " magnitude, and the average of the z-transformed expression of ligand and"
        " receptor as a measure of specificity."
    ),
    paper_name=(
        "Computation and visualization of cell–cell signaling topologies in single-cell"
        " systems data using Connectome"
    ),
    paper_reference="raredon2022computation",
    paper_year=2022,
)


def _connectome(adata, test=False):
    return _liana(adata=adata, method=connectome, test=test, score_key=connectome.specificity)


@_connectome_method(
    method_name="Connectome (max)",
)
def connectome_max(adata, test=False):
    adata = _connectome(adata, test=test)
    adata.uns["ccc_pred"] = aggregate_method_scores(adata, how="max")

    return adata


@_connectome_method(
    method_name="Connectome (sum)",
)
def connectome_sum(adata, test=False):
    adata = _connectome(adata, test=test)
    adata.uns["ccc_pred"] = aggregate_method_scores(adata, how="sum")

    return adata


_logfc_method = functools.partial(
    _liana_method,
    method_summary=(
        "logFC (implemented in LIANA and inspired by iTALK) combines both expression"
        " and magnitude, and represents the average of one-versus-the-rest log2-fold"
        " change of ligand and receptor expression per cell type."
    ),
)


def _logfc(adata, test=False):
    return _liana(adata=adata, method=logfc, test=test, score_key=logfc.specificity)


@_logfc_method(
    method_name="Log2FC (max)",
)
def logfc_max(adata, test=False):
    adata = _logfc(adata, test=test)
    adata.uns["ccc_pred"] = aggregate_method_scores(adata, how="max")

    return adata


@_logfc_method(
    method_name="Log2FC (sum)",
)
def logfc_sum(adata, test=False):
    adata = _logfc(adata, test=test)
    adata.uns["ccc_pred"] = aggregate_method_scores(adata, how="sum")

    return adata


_natmi_method = functools.partial(
    _liana_method,
    method_summary=(
        "NATMI uses the product of ligand-receptor expression as a measure of"
        " magnitude. As a measure of specificity, NATMI proposes $specificity.edge ="
        r" \frac{l}{l_s} \cdot \frac{r}{r_s}$; where $l$ and $r$ represent the average"
        " expression of ligand and receptor per cell type, and $l_s$ and $r_s$"
        " represent the sums of the average ligand and receptor expression across all"
        " cell types. We use its specificity measure, as recommended by the authors for"
        " single-context predictions."
    ),
    paper_name="Predicting cell-to-cell communication networks using NATMI",
    paper_reference="hou2020predicting",
    paper_year=2021,
)


def _natmi(adata, test=False):
    return _liana(adata=adata, method=natmi, test=test, score_key=natmi.specificity)


@_natmi_method(
    method_name="NATMI (max)",
)
def natmi_max(adata, test=False):
    adata = _natmi(adata, test=test)
    adata.uns["ccc_pred"] = aggregate_method_scores(adata, how="max")

    return adata


@_natmi_method(
    method_name="NATMI (sum)",
)
def natmi_sum(adata, test=False):
    adata = _natmi(adata, test=test)
    adata.uns["ccc_pred"] = aggregate_method_scores(adata, how="sum")

    return adata


_sca_method = functools.partial(
    _liana_method,
    method_summary=(
        "SingleCellSignalR provides a magnitude score as $LRscore ="
        r" \frac{\sqrt{lr}}{\mu+\sqrt{lr}}$; where $l$ and $r$ are the average ligand"
        r" and receptor expression per cell type, and $\mu$ is the mean of the"
        " expression matrix."
    ),
    paper_name=(
        "SingleCellSignalR: inference of intercellular networks from single-cell"
        " transcriptomics"
    ),
    paper_reference="cabello2020singlecellsignalr",
    paper_year=2021,
)


def _sca(adata, test=False):
    return _liana(adata=adata, method=singlecellsignalr, test=test, score_key=singlecellsignalr.magnitude)


@_sca_method(
    method_name="SingleCellSignalR (max)",
)
def sca_max(adata, test=False):
    adata = _sca(adata, test=test)
    adata.uns["ccc_pred"] = aggregate_method_scores(adata, how="max")

    return adata


@_sca_method(
    method_name="SingleCellSignalR (sum)",
)
def sca_sum(adata, test=False):
    adata = _sca(adata, test=test)
    adata.uns["ccc_pred"] = aggregate_method_scores(adata, how="sum")

    return adata

_cellchat_method = functools.partial(
    _liana_method,
    method_summary=(
        "This a resource-agnostic adaptation of CellChat, provides a magnitude score as:"
        r"$ LRprob = \frac{TriMean(L) \cdot TriMean(R)}{Kh + TriMean(L) \cdot TriMean(R)} $"
        " where Kh = 0.5 by default and `TriMean` represents Tuckey's Trimean function:"
        r" $TriMean(X) = \frac{Q_{0.25}(X) + 2 \cdot Q_{0.5}(X) + Q_{0.75}(X)}{4}$"
        " CellChat also provides a permutation-based p-value as a measure of specificity."
        " Here, we use the former to prioritize interactions, subsequent"
        " to filtering according to p-value less than 0.05."
    ),
    paper_name=(
        "Inference and analysis of cell-cell communication using CellChat"
    ),
    paper_reference="jin2022cellchat",
    paper_year=2022,
)

def _cellchat(adata, test=False):
    return _liana(adata=adata, method=cellchat, test=test, score_key=cellchat.magnitude)


@_cellchat_method(
    method_name="CellChat (sum)",
)
def cellchat_max(adata, test=False):
    adata = _cellchat(adata, test=test)
    adata.uns["ccc_pred"] = aggregate_method_scores(adata, how="sum")

    return adata


@_cellchat_method(
    method_name="CellChat (max)",
)
def cellchat_max(adata, test=False):
    adata = _cellchat(adata, test=test)
    adata.uns["ccc_pred"] = aggregate_method_scores(adata, how="max")

    return adata


_geometric_mean_method = functools.partial(
    _liana_method,
    method_summary=(
        "Equivalent to CellPhoneDBv2 method but it calculates a geometric mean of"
        " ligand-receptor expression as a measure of interaction magnitude,"
        " along with a permutation-based p-value as a measure of specificity."
        " Here, we use the former to prioritize interactions, subsequent"
        " to filtering according to p-value less than 0.05."
    ),
    paper_name=( # TODO change to liana x Tensor paper? Or just keep it as is?
        "CellPhoneDB: inferring cell–cell communication from combined expression of"
        " multi-subunit ligand–receptor complexes"
    ),
    paper_reference="efremova2020cellphonedb",
    paper_year=2020,
)

def _geomeric_mean(adata, test=False):
    return _liana(adata=adata, method=geometric_mean, test=test, score_key=geometric_mean.magnitude)


@_geometric_mean_method(
    method_name="Geometric mean (sum)",
)
def cellchat_max(adata, test=False):
    adata = _geomeric_mean(adata, test=test)
    adata.uns["ccc_pred"] = aggregate_method_scores(adata, how="sum")

    return adata


@_geometric_mean_method(
    method_name="Geometric mean (max)",
)
def cellchat_max(adata, test=False):
    adata = _geomeric_mean(adata, test=test)
    adata.uns["ccc_pred"] = aggregate_method_scores(adata, how="max")

    return adata
