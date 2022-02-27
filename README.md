# Seismic Facies Segmentation

Project about Semantic Segmentation in Seismic Images (Facies).


## Objective

Implement deep neural networks for Semantic Segmentation for Facies Clasification. 

## Dataset

The dataset used was the Netherlands F3 block, which is a fully-annotated 3D geological model open-sourced by [Alaudah et al](https://arxiv.org/pdf/1901.07659.pdf). It has six classes, where each one represents a facies with the exception of one that is the union of two facies because it was difficult to define the limits between them. The three-dimensional block has a dimension of 600x900x255. In order to get a model that generalizes correctly, ranges were defined to split the data in one block for training and two testing blocks. 

## Implementation

Docker available for this proyect:
```
docker pull smitharauco/tensorflow_1.13:latest
```
First, .txt files are generated for loading the sections after performing the split in training, validation and test set. For this use the notebook `generate_and_load_sections.ipynb`.

For training the model, you can see an example in `Baseline_UNet_Train.ipynb`.

## Results

The following table shows the two best results, which managed to outperform the results of the paper that presented the data. For more details, see [Towards a Benchmark for Sedimentary Facies Classification: Applied to the Netherlands F3 Block](https://www.researchgate.net/publication/351508702_Towards_a_Benchmark_for_Sedimentary_Facies_Classification_Applied_to_the_Netherlands_F3_Block).

| Model      | Pixel Accuracy| Mean Class Accuracy| Frequency-Weighted Intersection over Union|
| ----------- | :----: | :----: | :----: |
| Alaudah et al.      | 0.905 | 0.817 | 0.832 |
| BiAtrousUNetConvLSTM      | 0.942 | 0.848 | 0.894 |
| Atrous UNet   | 0.943 | 0.871 | 0.895 |

## References

```tex
@InProceedings{campos2020f3,
    author="Campos Trinidad, Maykol J. and Arauco Canchumuni, Smith W. and Cavalcanti Pacheco, Marco Aurelio",
    title="Towards a Benchmark for Sedimentary Facies Classification: Applied to the Netherlands F3 Block",
    booktitle="Information Management and Big Data",
    year="2021",
    publisher="Springer International Publishing",
    address="Cham",
    pages="211--222",
    isbn="978-3-030-76228-5"
}
```
https://link.springer.com/chapter/10.1007/978-3-030-76228-5_15



```tex
@InProceedings{campos2021convlstmf3,
    author="Campos Trinidad, Maykol J. and Arauco Canchumuni, Smith W., Queiroz Feitosa, Raul and Cavalcanti Pacheco, Marco Aurélio",
    title="Seismic Facies Segmentation Using Atrous Convolutional-LSTM Network",
    year="2021",
    conference="Proceedings of the XLII Ibero-Latin-American Congress on Computational Methods in Engineering and III Pan-American Congress on Computational Mechanics, ABMEC-IACM",
}
```

https://cilamce.com.br/anais/arearestrita/apresentacoes/252/10005.pdf