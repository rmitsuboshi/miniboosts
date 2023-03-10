# `miniboosts/src/booster` directory

This directory defines boosting algorithms.
The boosting algorithms defined in this directory are listed below:


### Boosting algorithms
* [AdaBoost](https://www.sciencedirect.com/science/article/pii/S002200009791504X?via%3Dihub) by Freund and Schapire, 1997.  
    AdaBoost is defined in `adaboost.rs`.
* [LPBoost](https://link.springer.com/content/pdf/10.1023/A:1012470815092.pdf) by Demiriz, Bennett, and Shawe-Taylor, 2002.  
    LPBoost is defined in `lpboost/`.
* [SmoothBoost](https://link.springer.com/chapter/10.1007/3-540-44581-1_31) by Rocco A. Servedio, 2003.  
    SmoothBoost is defined in `smoothboost` directory.
* [AdaBoostV](http://jmlr.org/papers/v6/ratsch05a.html) by Rätsch and Warmuth, 2005.  
    AdaBoostV is defined in `adaboostv.rs`.
* [TotalBoost](https://dl.acm.org/doi/10.1145/1143844.1143970) by Warmuth, Liao, and Rätsch, 2006.  
    TotalBoost is defined in `totalboost.rs`.
* [SoftBoost](https://proceedings.neurips.cc/paper/2007/file/cfbce4c1d7c425baf21d6b6f2babe6be-Paper.pdf) by Warmuth, Glocer, and Rätsch, 2007.  
    SoftBoost is defined in `softboost.rs`.
* [ERLPBoost](https://www.stat.purdue.edu/~vishy/papers/WarGloVis08.pdf) by Warmuth and Glocer, and Vishwanathan, 2008.  
    ERLPBoost is defined in `erlpboost/` directory.
* [CERLPBoost](https://link.springer.com/article/10.1007/s10994-010-5173-z) (The Corrective ERLPBoost) by Shalev-Shwartz and Singer, 2010.  
    CERLPBoost is defined in `cerlpboost/` directory.
* [MLPBoost](https://arxiv.org/abs/2209.10831) by Mitsuboshi, Hatano, and Takimoto, 2022.  
    MLPBoost is defined in `mlpboost/` directory.


### `Booster` trait
`core.rs` defines `Booster` trait and `State` struct.
If you want to implement your own boosting algorithm,
you must implement `Booster` trait.

See the doc string for further information.


### Directory structure

```txt
./
├─ core.rs         Defines Booster trait
│
├─ adaboost
│  └ adaboost.rs    Defines AdaBoost
├─ adaboostv
│  └ adaboostv.rs   Defines AdaBoost*
├─ cerlpboost
│  └ cerlpboost.rs  Defines Corrective ERLPBoost
├─ erlpboost
│  ├ qp_model.rs    Implements the sub-problem (QP) for ERLPBoost
│  └ erlpboost.rs   Defines ERLPBoost
├─ gradient_boost
│  └ gbm.rs         Defines Gradient Boosting Machine
├─ lpboost
│  ├ lp_model.rs    Implements the sub-problem (LP) for LPBoost
│  └ lpboost.rs     Defines LPBoost
├─ mlpboost
│  ├ dist.rs        Implements a function that returns a distribution on examples
│  ├ options.rs     Implements some optional behavors for MLPBoost
│  ├ utils.rs       Implements some utility function
│  ├ lp_model.rs    Implements the sub-problem (LP) for MLPBoost
│  └ mlpboost.rs    Defines MLPBoost
├─ smoothboost
│  └ smoothboost.rs Defines SmoothBoost
├─ softboost
│  └ softboost.rs   Defines SoftBoost
└─ totalboost
   └ totalboost.rs  Defines TotalBoost
```
