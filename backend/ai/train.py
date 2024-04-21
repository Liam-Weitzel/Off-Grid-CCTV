from ultralytics import YOLO
import fasterrcnn_resnet50_fpn
import maskrcnn_resnet50_fpn
import retinanet_resnet50_fpn

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                                TRAIN_IMAGES="./data/duomo/train/images",
                                                VAL_IMAGES="./data/duomo/valid/images",
                                                VAL_ANNOTATIONS="./data/duomo/valid/images",
                                                IMG_WIDTH=640,
                                                EPOCHS=100,
                                                BATCH_SIZE=2,
                                                num_classes=2,
                                                label_to_index={'person': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                                TRAIN_IMAGES="./data/duomo/train/images",
                                                VAL_IMAGES="./data/duomo/valid/images",
                                                VAL_ANNOTATIONS="./data/duomo/valid/images",
                                                IMG_WIDTH=640,
                                                EPOCHS=200,
                                                BATCH_SIZE=2,
                                                num_classes=2,
                                                label_to_index={'person': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                                TRAIN_IMAGES="./data/duomo/train/images",
                                                VAL_IMAGES="./data/duomo/valid/images",
                                                VAL_ANNOTATIONS="./data/duomo/valid/images",
                                                IMG_WIDTH=640,
                                                EPOCHS=300,
                                                BATCH_SIZE=2,
                                                num_classes=2,
                                                label_to_index={'person': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                                TRAIN_IMAGES="./data/duomo/train/images",
                                                VAL_IMAGES="./data/duomo/valid/images",
                                                VAL_ANNOTATIONS="./data/duomo/valid/images",
                                                IMG_WIDTH=640,
                                                EPOCHS=400,
                                                BATCH_SIZE=2,
                                                num_classes=2,
                                                label_to_index={'person': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                                TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                                VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                                VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/images",
                                                IMG_WIDTH=320,
                                                EPOCHS=100,
                                                BATCH_SIZE=2,
                                                num_classes=2,
                                                label_to_index={'person': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                                TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                                VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                                VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/images",
                                                IMG_WIDTH=320,
                                                EPOCHS=200,
                                                BATCH_SIZE=2,
                                                num_classes=2,
                                                label_to_index={'person': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                                TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                                VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                                VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/images",
                                                IMG_WIDTH=320,
                                                EPOCHS=300,
                                                BATCH_SIZE=2,
                                                num_classes=2,
                                                label_to_index={'person': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                                TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                                VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                                VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/images",
                                                IMG_WIDTH=320,
                                                EPOCHS=400,
                                                BATCH_SIZE=2,
                                                num_classes=2,
                                                label_to_index={'person': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                                TRAIN_IMAGES="./data/keskvaljak/train/images",
                                                VAL_IMAGES="./data/keskvaljak/valid/images",
                                                VAL_ANNOTATIONS="./data/keskvaljak/valid/images",
                                                IMG_WIDTH=320,
                                                EPOCHS=100,
                                                BATCH_SIZE=2,
                                                num_classes=4,
                                                label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                                TRAIN_IMAGES="./data/keskvaljak/train/images",
                                                VAL_IMAGES="./data/keskvaljak/valid/images",
                                                VAL_ANNOTATIONS="./data/keskvaljak/valid/images",
                                                IMG_WIDTH=320,
                                                EPOCHS=200,
                                                BATCH_SIZE=2,
                                                num_classes=4,
                                                label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                                TRAIN_IMAGES="./data/keskvaljak/train/images",
                                                VAL_IMAGES="./data/keskvaljak/valid/images",
                                                VAL_ANNOTATIONS="./data/keskvaljak/valid/images",
                                                IMG_WIDTH=320,
                                                EPOCHS=300,
                                                BATCH_SIZE=2,
                                                num_classes=4,
                                                label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                                TRAIN_IMAGES="./data/keskvaljak/train/images",
                                                VAL_IMAGES="./data/keskvaljak/valid/images",
                                                VAL_ANNOTATIONS="./data/keskvaljak/valid/images",
                                                IMG_WIDTH=320,
                                                EPOCHS=400,
                                                BATCH_SIZE=2,
                                                num_classes=4,
                                                label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                                TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                                VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                                VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/images",
                                                IMG_WIDTH=320,
                                                EPOCHS=100,
                                                BATCH_SIZE=2,
                                                num_classes=3,
                                                label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                                TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                                VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                                VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/images",
                                                IMG_WIDTH=320,
                                                EPOCHS=200,
                                                BATCH_SIZE=2,
                                                num_classes=3,
                                                label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                                TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                                VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                                VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/images",
                                                IMG_WIDTH=320,
                                                EPOCHS=300,
                                                BATCH_SIZE=2,
                                                num_classes=3,
                                                label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                                TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                                VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                                VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/images",
                                                IMG_WIDTH=320,
                                                EPOCHS=400,
                                                BATCH_SIZE=2,
                                                num_classes=3,
                                                label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                                TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                                VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                                VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/images",
                                                IMG_WIDTH=480,
                                                EPOCHS=100,
                                                BATCH_SIZE=2,
                                                num_classes=3,
                                                label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                                TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                                VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                                VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/images",
                                                IMG_WIDTH=480,
                                                EPOCHS=200,
                                                BATCH_SIZE=2,
                                                num_classes=3,
                                                label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                                TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                                VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                                VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/images",
                                                IMG_WIDTH=480,
                                                EPOCHS=300,
                                                BATCH_SIZE=2,
                                                num_classes=3,
                                                label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                                TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                                VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                                VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/images",
                                                IMG_WIDTH=480,
                                                EPOCHS=400,
                                                BATCH_SIZE=2,
                                                num_classes=3,
                                                label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                            TRAIN_IMAGES="./data/duomo/train/images",
                                            VAL_IMAGES="./data/duomo/valid/images",
                                            VAL_ANNOTATIONS="./data/duomo/valid/images",
                                            IMG_WIDTH=640,
                                            EPOCHS=100,
                                            BATCH_SIZE=2,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                            TRAIN_IMAGES="./data/duomo/train/images",
                                            VAL_IMAGES="./data/duomo/valid/images",
                                            VAL_ANNOTATIONS="./data/duomo/valid/images",
                                            IMG_WIDTH=640,
                                            EPOCHS=200,
                                            BATCH_SIZE=2,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                            TRAIN_IMAGES="./data/duomo/train/images",
                                            VAL_IMAGES="./data/duomo/valid/images",
                                            VAL_ANNOTATIONS="./data/duomo/valid/images",
                                            IMG_WIDTH=640,
                                            EPOCHS=300,
                                            BATCH_SIZE=2,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                            TRAIN_IMAGES="./data/duomo/train/images",
                                            VAL_IMAGES="./data/duomo/valid/images",
                                            VAL_ANNOTATIONS="./data/duomo/valid/images",
                                            IMG_WIDTH=640,
                                            EPOCHS=400,
                                            BATCH_SIZE=2,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                            TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                            VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                            VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=100,
                                            BATCH_SIZE=2,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                            TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                            VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                            VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=200,
                                            BATCH_SIZE=2,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                            TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                            VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                            VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=300,
                                            BATCH_SIZE=2,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                            TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                            VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                            VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=400,
                                            BATCH_SIZE=2,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                            TRAIN_IMAGES="./data/keskvaljak/train/images",
                                            VAL_IMAGES="./data/keskvaljak/valid/images",
                                            VAL_ANNOTATIONS="./data/keskvaljak/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=100,
                                            BATCH_SIZE=2,
                                            num_classes=4,
                                            label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                            TRAIN_IMAGES="./data/keskvaljak/train/images",
                                            VAL_IMAGES="./data/keskvaljak/valid/images",
                                            VAL_ANNOTATIONS="./data/keskvaljak/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=200,
                                            BATCH_SIZE=2,
                                            num_classes=4,
                                            label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                            TRAIN_IMAGES="./data/keskvaljak/train/images",
                                            VAL_IMAGES="./data/keskvaljak/valid/images",
                                            VAL_ANNOTATIONS="./data/keskvaljak/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=300,
                                            BATCH_SIZE=2,
                                            num_classes=4,
                                            label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                            TRAIN_IMAGES="./data/keskvaljak/train/images",
                                            VAL_IMAGES="./data/keskvaljak/valid/images",
                                            VAL_ANNOTATIONS="./data/keskvaljak/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=400,
                                            BATCH_SIZE=2,
                                            num_classes=4,
                                            label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                            TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                            VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                            VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=100,
                                            BATCH_SIZE=2,
                                            num_classes=3,
                                            label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                            TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                            VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                            VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=200,
                                            BATCH_SIZE=2,
                                            num_classes=3,
                                            label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                            TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                            VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                            VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=300,
                                            BATCH_SIZE=2,
                                            num_classes=3,
                                            label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                            TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                            VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                            VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=400,
                                            BATCH_SIZE=2,
                                            num_classes=3,
                                            label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                            TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                            VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                            VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/images",
                                            IMG_WIDTH=480,
                                            EPOCHS=100,
                                            BATCH_SIZE=2,
                                            num_classes=3,
                                            label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                            TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                            VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                            VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/images",
                                            IMG_WIDTH=480,
                                            EPOCHS=200,
                                            BATCH_SIZE=2,
                                            num_classes=3,
                                            label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                            TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                            VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                            VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/images",
                                            IMG_WIDTH=480,
                                            EPOCHS=300,
                                            BATCH_SIZE=2,
                                            num_classes=3,
                                            label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                            TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                            VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                            VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/images",
                                            IMG_WIDTH=480,
                                            EPOCHS=400,
                                            BATCH_SIZE=2,
                                            num_classes=3,
                                            label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                            TRAIN_IMAGES="./data/duomo/train/images",
                                            VAL_IMAGES="./data/duomo/valid/images",
                                            VAL_ANNOTATIONS="./data/duomo/valid/images",
                                            IMG_WIDTH=640,
                                            EPOCHS=100,
                                            BATCH_SIZE=2,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                            TRAIN_IMAGES="./data/duomo/train/images",
                                            VAL_IMAGES="./data/duomo/valid/images",
                                            VAL_ANNOTATIONS="./data/duomo/valid/images",
                                            IMG_WIDTH=640,
                                            EPOCHS=200,
                                            BATCH_SIZE=2,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                            TRAIN_IMAGES="./data/duomo/train/images",
                                            VAL_IMAGES="./data/duomo/valid/images",
                                            VAL_ANNOTATIONS="./data/duomo/valid/images",
                                            IMG_WIDTH=640,
                                            EPOCHS=300,
                                            BATCH_SIZE=2,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                            TRAIN_IMAGES="./data/duomo/train/images",
                                            VAL_IMAGES="./data/duomo/valid/images",
                                            VAL_ANNOTATIONS="./data/duomo/valid/images",
                                            IMG_WIDTH=640,
                                            EPOCHS=400,
                                            BATCH_SIZE=2,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                            TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                            VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                            VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=100,
                                            BATCH_SIZE=2,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                            TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                            VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                            VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=200,
                                            BATCH_SIZE=2,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                            TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                            VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                            VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=300,
                                            BATCH_SIZE=2,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                            TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                            VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                            VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=400,
                                            BATCH_SIZE=2,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                            TRAIN_IMAGES="./data/keskvaljak/train/images",
                                            VAL_IMAGES="./data/keskvaljak/valid/images",
                                            VAL_ANNOTATIONS="./data/keskvaljak/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=100,
                                            BATCH_SIZE=2,
                                            num_classes=4,
                                            label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                            TRAIN_IMAGES="./data/keskvaljak/train/images",
                                            VAL_IMAGES="./data/keskvaljak/valid/images",
                                            VAL_ANNOTATIONS="./data/keskvaljak/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=200,
                                            BATCH_SIZE=2,
                                            num_classes=4,
                                            label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                            TRAIN_IMAGES="./data/keskvaljak/train/images",
                                            VAL_IMAGES="./data/keskvaljak/valid/images",
                                            VAL_ANNOTATIONS="./data/keskvaljak/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=300,
                                            BATCH_SIZE=2,
                                            num_classes=4,
                                            label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                            TRAIN_IMAGES="./data/keskvaljak/train/images",
                                            VAL_IMAGES="./data/keskvaljak/valid/images",
                                            VAL_ANNOTATIONS="./data/keskvaljak/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=400,
                                            BATCH_SIZE=2,
                                            num_classes=4,
                                            label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                            TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                            VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                            VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=100,
                                            BATCH_SIZE=2,
                                            num_classes=3,
                                            label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                            TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                            VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                            VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=200,
                                            BATCH_SIZE=2,
                                            num_classes=3,
                                            label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                            TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                            VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                            VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=300,
                                            BATCH_SIZE=2,
                                            num_classes=3,
                                            label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                            TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                            VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                            VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=400,
                                            BATCH_SIZE=2,
                                            num_classes=3,
                                            label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                            TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                            VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                            VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/images",
                                            IMG_WIDTH=480,
                                            EPOCHS=100,
                                            BATCH_SIZE=2,
                                            num_classes=3,
                                            label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                            TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                            VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                            VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/images",
                                            IMG_WIDTH=480,
                                            EPOCHS=200,
                                            BATCH_SIZE=2,
                                            num_classes=3,
                                            label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                            TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                            VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                            VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/images",
                                            IMG_WIDTH=480,
                                            EPOCHS=300,
                                            BATCH_SIZE=2,
                                            num_classes=3,
                                            label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                            TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                            VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                            VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/images",
                                            IMG_WIDTH=480,
                                            EPOCHS=400,
                                            BATCH_SIZE=2,
                                            num_classes=3,
                                            label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

# PRETRAINED ^^

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                                TRAIN_IMAGES="./data/duomo/train/images",
                                                VAL_IMAGES="./data/duomo/valid/images",
                                                VAL_ANNOTATIONS="./data/duomo/valid/images",
                                                IMG_WIDTH=640,
                                                EPOCHS=100,
                                                BATCH_SIZE=2,
                                                num_classes=2,
                                                label_to_index={'person': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                                TRAIN_IMAGES="./data/duomo/train/images",
                                                VAL_IMAGES="./data/duomo/valid/images",
                                                VAL_ANNOTATIONS="./data/duomo/valid/images",
                                                IMG_WIDTH=640,
                                                EPOCHS=200,
                                                BATCH_SIZE=2,
                                                num_classes=2,
                                                label_to_index={'person': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                                TRAIN_IMAGES="./data/duomo/train/images",
                                                VAL_IMAGES="./data/duomo/valid/images",
                                                VAL_ANNOTATIONS="./data/duomo/valid/images",
                                                IMG_WIDTH=640,
                                                EPOCHS=300,
                                                BATCH_SIZE=2,
                                                num_classes=2,
                                                label_to_index={'person': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                                TRAIN_IMAGES="./data/duomo/train/images",
                                                VAL_IMAGES="./data/duomo/valid/images",
                                                VAL_ANNOTATIONS="./data/duomo/valid/images",
                                                IMG_WIDTH=640,
                                                EPOCHS=400,
                                                BATCH_SIZE=2,
                                                num_classes=2,
                                                label_to_index={'person': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                                TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                                VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                                VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/images",
                                                IMG_WIDTH=320,
                                                EPOCHS=100,
                                                BATCH_SIZE=2,
                                                num_classes=2,
                                                label_to_index={'person': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                                TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                                VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                                VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/images",
                                                IMG_WIDTH=320,
                                                EPOCHS=200,
                                                BATCH_SIZE=2,
                                                num_classes=2,
                                                label_to_index={'person': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                                TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                                VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                                VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/images",
                                                IMG_WIDTH=320,
                                                EPOCHS=300,
                                                BATCH_SIZE=2,
                                                num_classes=2,
                                                label_to_index={'person': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                                TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                                VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                                VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/images",
                                                IMG_WIDTH=320,
                                                EPOCHS=400,
                                                BATCH_SIZE=2,
                                                num_classes=2,
                                                label_to_index={'person': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                                TRAIN_IMAGES="./data/keskvaljak/train/images",
                                                VAL_IMAGES="./data/keskvaljak/valid/images",
                                                VAL_ANNOTATIONS="./data/keskvaljak/valid/images",
                                                IMG_WIDTH=320,
                                                EPOCHS=100,
                                                BATCH_SIZE=2,
                                                num_classes=4,
                                                label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                                TRAIN_IMAGES="./data/keskvaljak/train/images",
                                                VAL_IMAGES="./data/keskvaljak/valid/images",
                                                VAL_ANNOTATIONS="./data/keskvaljak/valid/images",
                                                IMG_WIDTH=320,
                                                EPOCHS=200,
                                                BATCH_SIZE=2,
                                                num_classes=4,
                                                label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                                TRAIN_IMAGES="./data/keskvaljak/train/images",
                                                VAL_IMAGES="./data/keskvaljak/valid/images",
                                                VAL_ANNOTATIONS="./data/keskvaljak/valid/images",
                                                IMG_WIDTH=320,
                                                EPOCHS=300,
                                                BATCH_SIZE=2,
                                                num_classes=4,
                                                label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                                TRAIN_IMAGES="./data/keskvaljak/train/images",
                                                VAL_IMAGES="./data/keskvaljak/valid/images",
                                                VAL_ANNOTATIONS="./data/keskvaljak/valid/images",
                                                IMG_WIDTH=320,
                                                EPOCHS=400,
                                                BATCH_SIZE=2,
                                                num_classes=4,
                                                label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                                TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                                VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                                VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/images",
                                                IMG_WIDTH=320,
                                                EPOCHS=100,
                                                BATCH_SIZE=2,
                                                num_classes=3,
                                                label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                                TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                                VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                                VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/images",
                                                IMG_WIDTH=320,
                                                EPOCHS=200,
                                                BATCH_SIZE=2,
                                                num_classes=3,
                                                label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                                TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                                VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                                VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/images",
                                                IMG_WIDTH=320,
                                                EPOCHS=300,
                                                BATCH_SIZE=2,
                                                num_classes=3,
                                                label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                                TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                                VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                                VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/images",
                                                IMG_WIDTH=320,
                                                EPOCHS=400,
                                                BATCH_SIZE=2,
                                                num_classes=3,
                                                label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                                TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                                VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                                VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/images",
                                                IMG_WIDTH=480,
                                                EPOCHS=100,
                                                BATCH_SIZE=2,
                                                num_classes=3,
                                                label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                                TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                                VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                                VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/images",
                                                IMG_WIDTH=480,
                                                EPOCHS=200,
                                                BATCH_SIZE=2,
                                                num_classes=3,
                                                label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                                TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                                VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                                VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/images",
                                                IMG_WIDTH=480,
                                                EPOCHS=300,
                                                BATCH_SIZE=2,
                                                num_classes=3,
                                                label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                                TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                                VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                                VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/images",
                                                IMG_WIDTH=480,
                                                EPOCHS=400,
                                                BATCH_SIZE=2,
                                                num_classes=3,
                                                label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                            TRAIN_IMAGES="./data/duomo/train/images",
                                            VAL_IMAGES="./data/duomo/valid/images",
                                            VAL_ANNOTATIONS="./data/duomo/valid/images",
                                            IMG_WIDTH=640,
                                            EPOCHS=100,
                                            BATCH_SIZE=2,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                            TRAIN_IMAGES="./data/duomo/train/images",
                                            VAL_IMAGES="./data/duomo/valid/images",
                                            VAL_ANNOTATIONS="./data/duomo/valid/images",
                                            IMG_WIDTH=640,
                                            EPOCHS=200,
                                            BATCH_SIZE=2,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                            TRAIN_IMAGES="./data/duomo/train/images",
                                            VAL_IMAGES="./data/duomo/valid/images",
                                            VAL_ANNOTATIONS="./data/duomo/valid/images",
                                            IMG_WIDTH=640,
                                            EPOCHS=300,
                                            BATCH_SIZE=2,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                            TRAIN_IMAGES="./data/duomo/train/images",
                                            VAL_IMAGES="./data/duomo/valid/images",
                                            VAL_ANNOTATIONS="./data/duomo/valid/images",
                                            IMG_WIDTH=640,
                                            EPOCHS=400,
                                            BATCH_SIZE=2,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                            TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                            VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                            VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=100,
                                            BATCH_SIZE=2,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                            TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                            VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                            VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=200,
                                            BATCH_SIZE=2,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                            TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                            VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                            VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=300,
                                            BATCH_SIZE=2,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                            TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                            VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                            VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=400,
                                            BATCH_SIZE=2,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                            TRAIN_IMAGES="./data/keskvaljak/train/images",
                                            VAL_IMAGES="./data/keskvaljak/valid/images",
                                            VAL_ANNOTATIONS="./data/keskvaljak/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=100,
                                            BATCH_SIZE=2,
                                            num_classes=4,
                                            label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                            TRAIN_IMAGES="./data/keskvaljak/train/images",
                                            VAL_IMAGES="./data/keskvaljak/valid/images",
                                            VAL_ANNOTATIONS="./data/keskvaljak/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=200,
                                            BATCH_SIZE=2,
                                            num_classes=4,
                                            label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                            TRAIN_IMAGES="./data/keskvaljak/train/images",
                                            VAL_IMAGES="./data/keskvaljak/valid/images",
                                            VAL_ANNOTATIONS="./data/keskvaljak/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=300,
                                            BATCH_SIZE=2,
                                            num_classes=4,
                                            label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                            TRAIN_IMAGES="./data/keskvaljak/train/images",
                                            VAL_IMAGES="./data/keskvaljak/valid/images",
                                            VAL_ANNOTATIONS="./data/keskvaljak/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=400,
                                            BATCH_SIZE=2,
                                            num_classes=4,
                                            label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                            TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                            VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                            VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=100,
                                            BATCH_SIZE=2,
                                            num_classes=3,
                                            label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                            TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                            VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                            VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=200,
                                            BATCH_SIZE=2,
                                            num_classes=3,
                                            label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                            TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                            VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                            VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=300,
                                            BATCH_SIZE=2,
                                            num_classes=3,
                                            label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                            TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                            VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                            VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=400,
                                            BATCH_SIZE=2,
                                            num_classes=3,
                                            label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                            TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                            VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                            VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/images",
                                            IMG_WIDTH=480,
                                            EPOCHS=100,
                                            BATCH_SIZE=2,
                                            num_classes=3,
                                            label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                            TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                            VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                            VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/images",
                                            IMG_WIDTH=480,
                                            EPOCHS=200,
                                            BATCH_SIZE=2,
                                            num_classes=3,
                                            label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                            TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                            VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                            VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/images",
                                            IMG_WIDTH=480,
                                            EPOCHS=300,
                                            BATCH_SIZE=2,
                                            num_classes=3,
                                            label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                            TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                            VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                            VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/images",
                                            IMG_WIDTH=480,
                                            EPOCHS=400,
                                            BATCH_SIZE=2,
                                            num_classes=3,
                                            label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                            TRAIN_IMAGES="./data/duomo/train/images",
                                            VAL_IMAGES="./data/duomo/valid/images",
                                            VAL_ANNOTATIONS="./data/duomo/valid/images",
                                            IMG_WIDTH=640,
                                            EPOCHS=100,
                                            BATCH_SIZE=2,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                            TRAIN_IMAGES="./data/duomo/train/images",
                                            VAL_IMAGES="./data/duomo/valid/images",
                                            VAL_ANNOTATIONS="./data/duomo/valid/images",
                                            IMG_WIDTH=640,
                                            EPOCHS=200,
                                            BATCH_SIZE=2,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                            TRAIN_IMAGES="./data/duomo/train/images",
                                            VAL_IMAGES="./data/duomo/valid/images",
                                            VAL_ANNOTATIONS="./data/duomo/valid/images",
                                            IMG_WIDTH=640,
                                            EPOCHS=300,
                                            BATCH_SIZE=2,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                            TRAIN_IMAGES="./data/duomo/train/images",
                                            VAL_IMAGES="./data/duomo/valid/images",
                                            VAL_ANNOTATIONS="./data/duomo/valid/images",
                                            IMG_WIDTH=640,
                                            EPOCHS=400,
                                            BATCH_SIZE=2,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                            TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                            VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                            VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=100,
                                            BATCH_SIZE=2,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                            TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                            VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                            VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=200,
                                            BATCH_SIZE=2,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                            TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                            VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                            VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=300,
                                            BATCH_SIZE=2,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                            TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                            VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                            VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=400,
                                            BATCH_SIZE=2,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                            TRAIN_IMAGES="./data/keskvaljak/train/images",
                                            VAL_IMAGES="./data/keskvaljak/valid/images",
                                            VAL_ANNOTATIONS="./data/keskvaljak/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=100,
                                            BATCH_SIZE=2,
                                            num_classes=4,
                                            label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                            TRAIN_IMAGES="./data/keskvaljak/train/images",
                                            VAL_IMAGES="./data/keskvaljak/valid/images",
                                            VAL_ANNOTATIONS="./data/keskvaljak/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=200,
                                            BATCH_SIZE=2,
                                            num_classes=4,
                                            label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                            TRAIN_IMAGES="./data/keskvaljak/train/images",
                                            VAL_IMAGES="./data/keskvaljak/valid/images",
                                            VAL_ANNOTATIONS="./data/keskvaljak/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=300,
                                            BATCH_SIZE=2,
                                            num_classes=4,
                                            label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                            TRAIN_IMAGES="./data/keskvaljak/train/images",
                                            VAL_IMAGES="./data/keskvaljak/valid/images",
                                            VAL_ANNOTATIONS="./data/keskvaljak/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=400,
                                            BATCH_SIZE=2,
                                            num_classes=4,
                                            label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                            TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                            VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                            VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=100,
                                            BATCH_SIZE=2,
                                            num_classes=3,
                                            label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                            TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                            VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                            VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=200,
                                            BATCH_SIZE=2,
                                            num_classes=3,
                                            label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                            TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                            VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                            VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=300,
                                            BATCH_SIZE=2,
                                            num_classes=3,
                                            label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                            TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                            VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                            VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/images",
                                            IMG_WIDTH=320,
                                            EPOCHS=400,
                                            BATCH_SIZE=2,
                                            num_classes=3,
                                            label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                            TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                            VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                            VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/images",
                                            IMG_WIDTH=480,
                                            EPOCHS=100,
                                            BATCH_SIZE=2,
                                            num_classes=3,
                                            label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                            TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                            VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                            VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/images",
                                            IMG_WIDTH=480,
                                            EPOCHS=200,
                                            BATCH_SIZE=2,
                                            num_classes=3,
                                            label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                            TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                            VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                            VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/images",
                                            IMG_WIDTH=480,
                                            EPOCHS=300,
                                            BATCH_SIZE=2,
                                            num_classes=3,
                                            label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                            TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                            VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                            VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/images",
                                            IMG_WIDTH=480,
                                            EPOCHS=400,
                                            BATCH_SIZE=2,
                                            num_classes=3,
                                            label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

model = YOLO('yolov8n.pt')
model.train(data='data/duomo/data.yaml', epochs=100, imgsz=640)

model = YOLO('yolov8n.pt')
model.train(data='data/duomo/data.yaml', epochs=200, imgsz=640)

model = YOLO('yolov8n.pt')
model.train(data='data/duomo/data.yaml', epochs=300, imgsz=640)

model = YOLO('yolov8n.pt')
model.train(data='data/duomo/data.yaml', epochs=400, imgsz=640)

model = YOLO('yolov8n.pt')
model.train(data='data/hadji_dimitar_square/data.yaml', epochs=100, imgsz=320)

model = YOLO('yolov8n.pt')
model.train(data='data/hadji_dimitar_square/data.yaml', epochs=200, imgsz=320)

model = YOLO('yolov8n.pt')
model.train(data='data/hadji_dimitar_square/data.yaml', epochs=300, imgsz=320)

model = YOLO('yolov8n.pt')
model.train(data='data/hadji_dimitar_square/data.yaml', epochs=400, imgsz=320)

model = YOLO('yolov8n.pt')
model.train(data='data/keskvaljak/data.yaml', epochs=100, imgsz=320)

model = YOLO('yolov8n.pt')
model.train(data='data/keskvaljak/data.yaml', epochs=200, imgsz=320)

model = YOLO('yolov8n.pt')
model.train(data='data/keskvaljak/data.yaml', epochs=300, imgsz=320)

model = YOLO('yolov8n.pt')
model.train(data='data/keskvaljak/data.yaml', epochs=400, imgsz=320)

model = YOLO('yolov8n.pt')
model.train(data='data/kielce_university_of_technology/data.yaml', epochs=100, imgsz=320)

model = YOLO('yolov8n.pt')
model.train(data='data/kielce_university_of_technology/data.yaml', epochs=200, imgsz=320)

model = YOLO('yolov8n.pt')
model.train(data='data/kielce_university_of_technology/data.yaml', epochs=300, imgsz=320)

model = YOLO('yolov8n.pt')
model.train(data='data/kielce_university_of_technology/data.yaml', epochs=400, imgsz=320)

model = YOLO('yolov8n.pt')
model.train(data='data/toggenburg_alpaca_ranch/data.yaml', epochs=100, imgsz=480)

model = YOLO('yolov8n.pt')
model.train(data='data/toggenburg_alpaca_ranch/data.yaml', epochs=200, imgsz=480)

model = YOLO('yolov8n.pt')
model.train(data='data/toggenburg_alpaca_ranch/data.yaml', epochs=300, imgsz=480)

model = YOLO('yolov8n.pt')
model.train(data='data/toggenburg_alpaca_ranch/data.yaml', epochs=400, imgsz=480)

model = YOLO('yolov8n.yaml')
model.train(data='data/duomo/data.yaml', epochs=100, imgsz=640)

model = YOLO('yolov8n.yaml')
model.train(data='data/duomo/data.yaml', epochs=200, imgsz=640)

model = YOLO('yolov8n.yaml')
model.train(data='data/duomo/data.yaml', epochs=300, imgsz=640)

model = YOLO('yolov8n.yaml')
model.train(data='data/duomo/data.yaml', epochs=400, imgsz=640)

model = YOLO('yolov8n.yaml')
model.train(data='data/hadji_dimitar_square/data.yaml', epochs=100, imgsz=320)

model = YOLO('yolov8n.yaml')
model.train(data='data/hadji_dimitar_square/data.yaml', epochs=200, imgsz=320)

model = YOLO('yolov8n.yaml')
model.train(data='data/hadji_dimitar_square/data.yaml', epochs=300, imgsz=320)

model = YOLO('yolov8n.yaml')
model.train(data='data/hadji_dimitar_square/data.yaml', epochs=400, imgsz=320)

model = YOLO('yolov8n.yaml')
model.train(data='data/keskvaljak/data.yaml', epochs=100, imgsz=320)

model = YOLO('yolov8n.yaml')
model.train(data='data/keskvaljak/data.yaml', epochs=200, imgsz=320)

model = YOLO('yolov8n.yaml')
model.train(data='data/keskvaljak/data.yaml', epochs=300, imgsz=320)

model = YOLO('yolov8n.yaml')
model.train(data='data/keskvaljak/data.yaml', epochs=400, imgsz=320)

model = YOLO('yolov8n.yaml')
model.train(data='data/kielce_university_of_technology/data.yaml', epochs=100, imgsz=320)

model = YOLO('yolov8n.yaml')
model.train(data='data/kielce_university_of_technology/data.yaml', epochs=200, imgsz=320)

model = YOLO('yolov8n.yaml')
model.train(data='data/kielce_university_of_technology/data.yaml', epochs=300, imgsz=320)

model = YOLO('yolov8n.yaml')
model.train(data='data/kielce_university_of_technology/data.yaml', epochs=400, imgsz=320)

model = YOLO('yolov8n.yaml')
model.train(data='data/toggenburg_alpaca_ranch/data.yaml', epochs=100, imgsz=480)

model = YOLO('yolov8n.yaml')
model.train(data='data/toggenburg_alpaca_ranch/data.yaml', epochs=200, imgsz=480)

model = YOLO('yolov8n.yaml')
model.train(data='data/toggenburg_alpaca_ranch/data.yaml', epochs=300, imgsz=480)

model = YOLO('yolov8n.yaml')
model.train(data='data/toggenburg_alpaca_ranch/data.yaml', epochs=400, imgsz=480)

#hyper parameter tuned training here (copy paste everything above and insert tuned hyper params)

model = YOLO('yolov8n.pt')
model.val(data='data/labeled/duomo/data.yaml', imgsz=640)

model = YOLO('yolov8n.pt')
model.val(data='data/hadji_dimitar_square/data.yaml', imgsz=320)

model = YOLO('yolov8n.pt')
model.val(data='data/keskvaljak/data.yaml', imgsz=320)

model = YOLO('yolov8n.pt')
model.val(data='data/kielce_university_of_technology/data.yaml', imgsz=320)

model = YOLO('yolov8n.pt')
model.val(data='data/toggenburg_alpaca_ranch/data.yaml', imgsz=480)
