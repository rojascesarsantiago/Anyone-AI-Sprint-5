# 1. Base Line Model

## Summary
This model was trained one time with standars parameters to serve as a base line model

## Results
- Minimum loss during validation: 3.47
- Accuracy on test: 0.21

## Specs
- Weigths: "imagenet"
- Learning rate: 0.001
- Epochs until minimum loss: 18
- Batch size: 32
- Drop out: 0.2
- Data augmentation: yes
    - Random flip: "horizontal_and_vertical"
    - Random rotation: 0.2
    - Random zoom:
        - Height factor: 0.2
        - Width factor: 0.2

# 2. Iteration over base line model

## Summary
This model was obteined by training over the baseline model trying different hyperparameters combinations; the best results are reported bellow. It's worth mentioning that the best results were achieved without retraining all the weights of the model (model.trainable = False).

## Results
- Minimum loss during validation: 3.08
- Accuracy on test: 0.28

## Specs
- Weigths: based on Baseline model and different iterations of training
- Learning rate: started with 0.005, and was reduced by *0.1 every 30 epochs
- Epochs until minimum loss: 76
- Batch size: 32
- Drop out: 0.2
- Data augmentation: yes
    - Random flip: "horizontal_and_vertical"
    - Random rotation: 0.2
    - Random zoom:
        - Height factor: 0.2
        - Width factor: 0.2

# 3. Model trained with remove background

## Summary
This model was trained with a dataset of car images cropped using Detectron2 to remove the background. This model was obteined by using the specs of the best iteration over the base line model.

## Results
- Minimum loss during validation: 2.36
- Accuracy on test: 0.42

## Specs
- Weigths: "imagenet"
- Learning rate: started with 0.005, and was reduced by *0.1 every 30 epochs
- Epochs until minimum loss: 43
- Batch size: 32
- Drop out: 0.2
- Data augmentation: yes
    - Random flip: "horizontal_and_vertical"
    - Random rotation: 0.2
    - Random zoom:
        - Height factor: 0.2
        - Width factor: 0.2
