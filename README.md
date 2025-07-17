# Solid-State Battery Performance Classification via Machine Learning

Author: lunazhang

This project presents a machine learning framework for the early prediction of performance and lifetime of solid-state lithium batteries. The research addresses the critical challenge of accelerating the development cycle of next-generation energy storage systems by moving beyond traditional, time-intensive experimental testing. By leveraging subtle electrochemical signatures from the initial formation cycles, our models classify cells into distinct performance tiers, enabling high-throughput screening of novel materials and manufacturing protocols.

本项目构建了一个基于机器学习的固态锂电池性能与寿命早期预测框架。该研究旨在解决下一代储能系统开发周期中的关键瓶颈——即如何超越传统耗时的实验测试流程。通过捕捉电池在初始活化循环中产生的细微电化学特征，我们建立的模型能够将电芯精确地分入不同的性能等级，从而实现对新材料和制造工艺的高通量筛选。

The advancement of solid-state batteries is hampered by complex degradation mechanisms, including lithium dendrite propagation, interfacial impedance growth, and chemo-mechanical failure. Characterizing these failure modes typically requires extensive cycling and post-mortem analysis. This work establishes a non-destructive, data-driven methodology to forecast these degradation trajectories. We hypothesize that the nascent electrochemical behavior contains sufficient information to predict long-term stability, thereby creating a feedback loop for rapid materials discovery and cell engineering optimization.

固态电池的发展受限于复杂的衰减机制，例如锂枝晶生长、界面阻抗增加以及化学-机械失效等。传统上，对这些失效模式的表征需要进行漫长的循环测试和复杂的物理拆解分析。本工作建立了一种无损的数据驱动方法，用以预测这些复杂的衰减路径。我们的核心假设是：电池早期的电化学行为蕴含了预测其长期稳定性的充分信息，从而为加速材料探索和电芯工程优化提供一个快速的反馈闭环。

Our pipeline integrates multi-modal data processing and advanced feature engineering. The input consists of high-precision chronopotentiometry and chronoamperometry data from hundreds of custom-fabricated Li|Garnet|NMC cells. We developed a feature library (`src/features_advanced.py`) comprising over 200 descriptors, including variance of the dQ/dV curve peaks, polarization voltage drift during GITT, and statistical moments of the coulombic efficiency decay profile. These features are designed to capture the subtle dynamics of interfacial kinetics and mass transport limitations. The classification task is performed using a Gradient Boosting Decision Tree (GBDT) model, selected for its interpretability and robustness with tabular data.

我们的技术管线集成了多模态数据处理与先进特征工程。输入数据源自数百个定制化 Li|石榴石|NMC 固态电芯的高精度恒流-恒压充放电测试。我们构建了一个包含超过200个描述符的特征库（`src/features_advanced.py`），这些特征包括dQ/dV曲线峰值的方差、GITT测试中的极化电压漂移以及库伦效率衰减曲线的统计矩等。这些特征旨在捕捉界面动力学和传质限制的细微动态。分类任务最终由梯度提升决策树（GBDT）模型执行，该模型因其对表格数据的强大鲁棒性和可解释性而被选用。

## Table of Contents
- [Repository Structure](#repository-structure)
- [Repository Usage](#repository-usage)

## Repository Structure

```
.
├── LICENSE
├── README.md
├── main.py
├── requirements.txt
└── src
    ├── __init__.py
    ├── data.py
    ├── data_pipeline.py
    ├── features.py
    ├── features_advanced.py
    ├── models.py
    └── train.py
```

## Repository Usage

### Environment Setup

This project requires Python 3.8+ and dependencies listed in `requirements.txt`.

本项目需要 Python 3.8+ 环境及 `requirements.txt` 文件中列出的依赖库。

```bash
# Clone the repository
git clone https://github.com/lunazhang/Battery_Performance_Classification-main.git
cd Battery_Performance_Classification-main

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### Execution

The main script `main.py` orchestrates the entire data processing and model training pipeline. It will execute data loading (`src/data.py`), feature extraction (`src/features.py`, `src/features_advanced.py`), model training (`src/train.py`), and save the results.

主脚本 `main.py` 驱动整个数据处理与模型训练流程。它将依次执行数据加载 (`src/data.py`)、特征提取 (`src/features.py`, `src/features_advanced.py`)、模型训练 (`src/train.py`) 并保存结果。

```bash
# Run the complete pipeline
python main.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

本项目基于 MIT 许可证开源 - 详细信息请查阅 [LICENSE](LICENSE) 文件。

