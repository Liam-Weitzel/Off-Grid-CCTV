from ultralytics import YOLO
import fasterrcnn_resnet50_fpn
import maskrcnn_resnet50_fpn
import retinanet_resnet50_fpn

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                                TRAIN_IMAGES="./data/duomo/train/images",
                                                VAL_IMAGES="./data/duomo/valid/images",
                                                VAL_ANNOTATIONS="./data/duomo/valid/xml",
                                                IMG_WIDTH=640,
                                                EPOCHS=50,
                                                BATCH_SIZE=7,
                                                num_classes=2,
                                                label_to_index={'person': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                                TRAIN_IMAGES="./data/duomo/train/images",
                                                VAL_IMAGES="./data/duomo/valid/images",
                                                VAL_ANNOTATIONS="./data/duomo/valid/xml",
                                                IMG_WIDTH=640,
                                                EPOCHS=100,
                                                BATCH_SIZE=7,
                                                num_classes=2,
                                                label_to_index={'person': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                                TRAIN_IMAGES="./data/duomo/train/images",
                                                VAL_IMAGES="./data/duomo/valid/images",
                                                VAL_ANNOTATIONS="./data/duomo/valid/xml",
                                                IMG_WIDTH=640,
                                                EPOCHS=150,
                                                BATCH_SIZE=7,
                                                num_classes=2,
                                                label_to_index={'person': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                                TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                                VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                                VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/xml",
                                                IMG_WIDTH=320,
                                                EPOCHS=50,
                                                BATCH_SIZE=7,
                                                num_classes=2,
                                                label_to_index={'person': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                                TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                                VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                                VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/xml",
                                                IMG_WIDTH=320,
                                                EPOCHS=100,
                                                BATCH_SIZE=7,
                                                num_classes=2,
                                                label_to_index={'person': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                                TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                                VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                                VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/xml",
                                                IMG_WIDTH=320,
                                                EPOCHS=150,
                                                BATCH_SIZE=7,
                                                num_classes=2,
                                                label_to_index={'person': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                                TRAIN_IMAGES="./data/keskvaljak/train/images",
                                                VAL_IMAGES="./data/keskvaljak/valid/images",
                                                VAL_ANNOTATIONS="./data/keskvaljak/valid/xml",
                                                IMG_WIDTH=320,
                                                EPOCHS=50,
                                                BATCH_SIZE=7,
                                                num_classes=4,
                                                label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                                TRAIN_IMAGES="./data/keskvaljak/train/images",
                                                VAL_IMAGES="./data/keskvaljak/valid/images",
                                                VAL_ANNOTATIONS="./data/keskvaljak/valid/xml",
                                                IMG_WIDTH=320,
                                                EPOCHS=100,
                                                BATCH_SIZE=7,
                                                num_classes=4,
                                                label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                                TRAIN_IMAGES="./data/keskvaljak/train/images",
                                                VAL_IMAGES="./data/keskvaljak/valid/images",
                                                VAL_ANNOTATIONS="./data/keskvaljak/valid/xml",
                                                IMG_WIDTH=320,
                                                EPOCHS=150,
                                                BATCH_SIZE=7,
                                                num_classes=4,
                                                label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                                TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                                VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                                VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/xml",
                                                IMG_WIDTH=320,
                                                EPOCHS=50,
                                                BATCH_SIZE=7,
                                                num_classes=3,
                                                label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                                TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                                VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                                VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/xml",
                                                IMG_WIDTH=320,
                                                EPOCHS=100,
                                                BATCH_SIZE=7,
                                                num_classes=3,
                                                label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                                TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                                VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                                VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/xml",
                                                IMG_WIDTH=320,
                                                EPOCHS=150,
                                                BATCH_SIZE=7,
                                                num_classes=3,
                                                label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                                TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                                VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                                VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/xml",
                                                IMG_WIDTH=480,
                                                EPOCHS=50,
                                                BATCH_SIZE=7,
                                                num_classes=3,
                                                label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                                TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                                VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                                VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/xml",
                                                IMG_WIDTH=480,
                                                EPOCHS=100,
                                                BATCH_SIZE=7,
                                                num_classes=3,
                                                label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                                TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                                VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                                VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/xml",
                                                IMG_WIDTH=480,
                                                EPOCHS=150,
                                                BATCH_SIZE=7,
                                                num_classes=3,
                                                label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                            TRAIN_IMAGES="./data/duomo/train/images",
                                            VAL_IMAGES="./data/duomo/valid/images",
                                            VAL_ANNOTATIONS="./data/duomo/valid/xml",
                                            IMG_WIDTH=640,
                                            EPOCHS=50,
                                            BATCH_SIZE=7,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                            TRAIN_IMAGES="./data/duomo/train/images",
                                            VAL_IMAGES="./data/duomo/valid/images",
                                            VAL_ANNOTATIONS="./data/duomo/valid/xml",
                                            IMG_WIDTH=640,
                                            EPOCHS=100,
                                            BATCH_SIZE=7,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                            TRAIN_IMAGES="./data/duomo/train/images",
                                            VAL_IMAGES="./data/duomo/valid/images",
                                            VAL_ANNOTATIONS="./data/duomo/valid/xml",
                                            IMG_WIDTH=640,
                                            EPOCHS=150,
                                            BATCH_SIZE=7,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                            TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                            VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                            VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=50,
                                            BATCH_SIZE=7,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                            TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                            VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                            VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=100,
                                            BATCH_SIZE=7,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                            TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                            VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                            VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=150,
                                            BATCH_SIZE=7,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                            TRAIN_IMAGES="./data/keskvaljak/train/images",
                                            VAL_IMAGES="./data/keskvaljak/valid/images",
                                            VAL_ANNOTATIONS="./data/keskvaljak/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=50,
                                            BATCH_SIZE=7,
                                            num_classes=4,
                                            label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                            TRAIN_IMAGES="./data/keskvaljak/train/images",
                                            VAL_IMAGES="./data/keskvaljak/valid/images",
                                            VAL_ANNOTATIONS="./data/keskvaljak/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=100,
                                            BATCH_SIZE=7,
                                            num_classes=4,
                                            label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                            TRAIN_IMAGES="./data/keskvaljak/train/images",
                                            VAL_IMAGES="./data/keskvaljak/valid/images",
                                            VAL_ANNOTATIONS="./data/keskvaljak/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=150,
                                            BATCH_SIZE=7,
                                            num_classes=4,
                                            label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                            TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                            VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                            VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=50,
                                            BATCH_SIZE=7,
                                            num_classes=3,
                                            label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                            TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                            VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                            VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=100,
                                            BATCH_SIZE=7,
                                            num_classes=3,
                                            label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                            TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                            VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                            VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=150,
                                            BATCH_SIZE=7,
                                            num_classes=3,
                                            label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                            TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                            VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                            VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/xml",
                                            IMG_WIDTH=480,
                                            EPOCHS=50,
                                            BATCH_SIZE=7,
                                            num_classes=3,
                                            label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                            TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                            VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                            VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/xml",
                                            IMG_WIDTH=480,
                                            EPOCHS=100,
                                            BATCH_SIZE=7,
                                            num_classes=3,
                                            label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                            TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                            VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                            VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/xml",
                                            IMG_WIDTH=480,
                                            EPOCHS=150,
                                            BATCH_SIZE=7,
                                            num_classes=3,
                                            label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                            TRAIN_IMAGES="./data/duomo/train/images",
                                            VAL_IMAGES="./data/duomo/valid/images",
                                            VAL_ANNOTATIONS="./data/duomo/valid/xml",
                                            IMG_WIDTH=640,
                                            EPOCHS=50,
                                            BATCH_SIZE=7,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                            TRAIN_IMAGES="./data/duomo/train/images",
                                            VAL_IMAGES="./data/duomo/valid/images",
                                            VAL_ANNOTATIONS="./data/duomo/valid/xml",
                                            IMG_WIDTH=640,
                                            EPOCHS=100,
                                            BATCH_SIZE=7,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                            TRAIN_IMAGES="./data/duomo/train/images",
                                            VAL_IMAGES="./data/duomo/valid/images",
                                            VAL_ANNOTATIONS="./data/duomo/valid/xml",
                                            IMG_WIDTH=640,
                                            EPOCHS=150,
                                            BATCH_SIZE=7,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                            TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                            VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                            VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=50,
                                            BATCH_SIZE=7,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                            TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                            VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                            VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=100,
                                            BATCH_SIZE=7,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                            TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                            VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                            VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=150,
                                            BATCH_SIZE=7,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                            TRAIN_IMAGES="./data/keskvaljak/train/images",
                                            VAL_IMAGES="./data/keskvaljak/valid/images",
                                            VAL_ANNOTATIONS="./data/keskvaljak/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=50,
                                            BATCH_SIZE=7,
                                            num_classes=4,
                                            label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                            TRAIN_IMAGES="./data/keskvaljak/train/images",
                                            VAL_IMAGES="./data/keskvaljak/valid/images",
                                            VAL_ANNOTATIONS="./data/keskvaljak/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=100,
                                            BATCH_SIZE=7,
                                            num_classes=4,
                                            label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                            TRAIN_IMAGES="./data/keskvaljak/train/images",
                                            VAL_IMAGES="./data/keskvaljak/valid/images",
                                            VAL_ANNOTATIONS="./data/keskvaljak/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=150,
                                            BATCH_SIZE=7,
                                            num_classes=4,
                                            label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                            TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                            VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                            VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=50,
                                            BATCH_SIZE=7,
                                            num_classes=3,
                                            label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                            TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                            VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                            VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=100,
                                            BATCH_SIZE=7,
                                            num_classes=3,
                                            label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                            TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                            VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                            VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=150,
                                            BATCH_SIZE=7,
                                            num_classes=3,
                                            label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                            TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                            VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                            VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/xml",
                                            IMG_WIDTH=480,
                                            EPOCHS=50,
                                            BATCH_SIZE=7,
                                            num_classes=3,
                                            label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                            TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                            VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                            VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/xml",
                                            IMG_WIDTH=480,
                                            EPOCHS=100,
                                            BATCH_SIZE=7,
                                            num_classes=3,
                                            label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                            TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                            VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                            VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/xml",
                                            IMG_WIDTH=480,
                                            EPOCHS=150,
                                            BATCH_SIZE=7,
                                            num_classes=3,
                                            label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=True)

# PRETRAINED ^^

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                                TRAIN_IMAGES="./data/duomo/train/images",
                                                VAL_IMAGES="./data/duomo/valid/images",
                                                VAL_ANNOTATIONS="./data/duomo/valid/xml",
                                                IMG_WIDTH=640,
                                                EPOCHS=50,
                                                BATCH_SIZE=7,
                                                num_classes=2,
                                                label_to_index={'person': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                                TRAIN_IMAGES="./data/duomo/train/images",
                                                VAL_IMAGES="./data/duomo/valid/images",
                                                VAL_ANNOTATIONS="./data/duomo/valid/xml",
                                                IMG_WIDTH=640,
                                                EPOCHS=100,
                                                BATCH_SIZE=7,
                                                num_classes=2,
                                                label_to_index={'person': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                                TRAIN_IMAGES="./data/duomo/train/images",
                                                VAL_IMAGES="./data/duomo/valid/images",
                                                VAL_ANNOTATIONS="./data/duomo/valid/xml",
                                                IMG_WIDTH=640,
                                                EPOCHS=150,
                                                BATCH_SIZE=7,
                                                num_classes=2,
                                                label_to_index={'person': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                                TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                                VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                                VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/xml",
                                                IMG_WIDTH=320,
                                                EPOCHS=50,
                                                BATCH_SIZE=7,
                                                num_classes=2,
                                                label_to_index={'person': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                                TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                                VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                                VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/xml",
                                                IMG_WIDTH=320,
                                                EPOCHS=100,
                                                BATCH_SIZE=7,
                                                num_classes=2,
                                                label_to_index={'person': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                                TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                                VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                                VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/xml",
                                                IMG_WIDTH=320,
                                                EPOCHS=150,
                                                BATCH_SIZE=7,
                                                num_classes=2,
                                                label_to_index={'person': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                                TRAIN_IMAGES="./data/keskvaljak/train/images",
                                                VAL_IMAGES="./data/keskvaljak/valid/images",
                                                VAL_ANNOTATIONS="./data/keskvaljak/valid/xml",
                                                IMG_WIDTH=320,
                                                EPOCHS=50,
                                                BATCH_SIZE=7,
                                                num_classes=4,
                                                label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                                TRAIN_IMAGES="./data/keskvaljak/train/images",
                                                VAL_IMAGES="./data/keskvaljak/valid/images",
                                                VAL_ANNOTATIONS="./data/keskvaljak/valid/xml",
                                                IMG_WIDTH=320,
                                                EPOCHS=100,
                                                BATCH_SIZE=7,
                                                num_classes=4,
                                                label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                                TRAIN_IMAGES="./data/keskvaljak/train/images",
                                                VAL_IMAGES="./data/keskvaljak/valid/images",
                                                VAL_ANNOTATIONS="./data/keskvaljak/valid/xml",
                                                IMG_WIDTH=320,
                                                EPOCHS=150,
                                                BATCH_SIZE=7,
                                                num_classes=4,
                                                label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                                TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                                VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                                VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/xml",
                                                IMG_WIDTH=320,
                                                EPOCHS=50,
                                                BATCH_SIZE=7,
                                                num_classes=3,
                                                label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                                TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                                VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                                VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/xml",
                                                IMG_WIDTH=320,
                                                EPOCHS=100,
                                                BATCH_SIZE=7,
                                                num_classes=3,
                                                label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                                TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                                VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                                VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/xml",
                                                IMG_WIDTH=320,
                                                EPOCHS=150,
                                                BATCH_SIZE=7,
                                                num_classes=3,
                                                label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                                TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                                VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                                VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/xml",
                                                IMG_WIDTH=480,
                                                EPOCHS=50,
                                                BATCH_SIZE=7,
                                                num_classes=3,
                                                label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                                TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                                VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                                VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/xml",
                                                IMG_WIDTH=480,
                                                EPOCHS=100,
                                                BATCH_SIZE=7,
                                                num_classes=3,
                                                label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

fasterrcnn_resnet50_fpn.fasterrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                                TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                                VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                                VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/xml",
                                                IMG_WIDTH=480,
                                                EPOCHS=150,
                                                BATCH_SIZE=7,
                                                num_classes=3,
                                                label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                                PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                            TRAIN_IMAGES="./data/duomo/train/images",
                                            VAL_IMAGES="./data/duomo/valid/images",
                                            VAL_ANNOTATIONS="./data/duomo/valid/xml",
                                            IMG_WIDTH=640,
                                            EPOCHS=50,
                                            BATCH_SIZE=7,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                            TRAIN_IMAGES="./data/duomo/train/images",
                                            VAL_IMAGES="./data/duomo/valid/images",
                                            VAL_ANNOTATIONS="./data/duomo/valid/xml",
                                            IMG_WIDTH=640,
                                            EPOCHS=100,
                                            BATCH_SIZE=7,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                            TRAIN_IMAGES="./data/duomo/train/images",
                                            VAL_IMAGES="./data/duomo/valid/images",
                                            VAL_ANNOTATIONS="./data/duomo/valid/xml",
                                            IMG_WIDTH=640,
                                            EPOCHS=150,
                                            BATCH_SIZE=7,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                            TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                            VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                            VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=50,
                                            BATCH_SIZE=7,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                            TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                            VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                            VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=100,
                                            BATCH_SIZE=7,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                            TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                            VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                            VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=150,
                                            BATCH_SIZE=7,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                            TRAIN_IMAGES="./data/keskvaljak/train/images",
                                            VAL_IMAGES="./data/keskvaljak/valid/images",
                                            VAL_ANNOTATIONS="./data/keskvaljak/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=50,
                                            BATCH_SIZE=7,
                                            num_classes=4,
                                            label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                            TRAIN_IMAGES="./data/keskvaljak/train/images",
                                            VAL_IMAGES="./data/keskvaljak/valid/images",
                                            VAL_ANNOTATIONS="./data/keskvaljak/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=100,
                                            BATCH_SIZE=7,
                                            num_classes=4,
                                            label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                            TRAIN_IMAGES="./data/keskvaljak/train/images",
                                            VAL_IMAGES="./data/keskvaljak/valid/images",
                                            VAL_ANNOTATIONS="./data/keskvaljak/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=150,
                                            BATCH_SIZE=7,
                                            num_classes=4,
                                            label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                            TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                            VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                            VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=50,
                                            BATCH_SIZE=7,
                                            num_classes=3,
                                            label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                            TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                            VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                            VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=100,
                                            BATCH_SIZE=7,
                                            num_classes=3,
                                            label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                            TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                            VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                            VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=150,
                                            BATCH_SIZE=7,
                                            num_classes=3,
                                            label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                            TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                            VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                            VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/xml",
                                            IMG_WIDTH=480,
                                            EPOCHS=50,
                                            BATCH_SIZE=7,
                                            num_classes=3,
                                            label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                            TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                            VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                            VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/xml",
                                            IMG_WIDTH=480,
                                            EPOCHS=100,
                                            BATCH_SIZE=7,
                                            num_classes=3,
                                            label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

maskrcnn_resnet50_fpn.maskrcnn_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                            TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                            VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                            VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/xml",
                                            IMG_WIDTH=480,
                                            EPOCHS=150,
                                            BATCH_SIZE=7,
                                            num_classes=3,
                                            label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                            TRAIN_IMAGES="./data/duomo/train/images",
                                            VAL_IMAGES="./data/duomo/valid/images",
                                            VAL_ANNOTATIONS="./data/duomo/valid/xml",
                                            IMG_WIDTH=640,
                                            EPOCHS=50,
                                            BATCH_SIZE=7,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                            TRAIN_IMAGES="./data/duomo/train/images",
                                            VAL_IMAGES="./data/duomo/valid/images",
                                            VAL_ANNOTATIONS="./data/duomo/valid/xml",
                                            IMG_WIDTH=640,
                                            EPOCHS=100,
                                            BATCH_SIZE=7,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/duomo/train/xml",
                                            TRAIN_IMAGES="./data/duomo/train/images",
                                            VAL_IMAGES="./data/duomo/valid/images",
                                            VAL_ANNOTATIONS="./data/duomo/valid/xml",
                                            IMG_WIDTH=640,
                                            EPOCHS=150,
                                            BATCH_SIZE=7,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                            TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                            VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                            VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=50,
                                            BATCH_SIZE=7,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                            TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                            VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                            VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=100,
                                            BATCH_SIZE=7,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/hadji_dimitar_square/train/xml",
                                            TRAIN_IMAGES="./data/hadji_dimitar_square/train/images",
                                            VAL_IMAGES="./data/hadji_dimitar_square/valid/images",
                                            VAL_ANNOTATIONS="./data/hadji_dimitar_square/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=150,
                                            BATCH_SIZE=7,
                                            num_classes=2,
                                            label_to_index={'person': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                            TRAIN_IMAGES="./data/keskvaljak/train/images",
                                            VAL_IMAGES="./data/keskvaljak/valid/images",
                                            VAL_ANNOTATIONS="./data/keskvaljak/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=50,
                                            BATCH_SIZE=7,
                                            num_classes=4,
                                            label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                            TRAIN_IMAGES="./data/keskvaljak/train/images",
                                            VAL_IMAGES="./data/keskvaljak/valid/images",
                                            VAL_ANNOTATIONS="./data/keskvaljak/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=100,
                                            BATCH_SIZE=7,
                                            num_classes=4,
                                            label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/keskvaljak/train/xml",
                                            TRAIN_IMAGES="./data/keskvaljak/train/images",
                                            VAL_IMAGES="./data/keskvaljak/valid/images",
                                            VAL_ANNOTATIONS="./data/keskvaljak/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=150,
                                            BATCH_SIZE=7,
                                            num_classes=4,
                                            label_to_index={'person': 3, 'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                            TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                            VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                            VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=50,
                                            BATCH_SIZE=7,
                                            num_classes=3,
                                            label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                            TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                            VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                            VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=100,
                                            BATCH_SIZE=7,
                                            num_classes=3,
                                            label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/kielce_university_of_technology/train/xml",
                                            TRAIN_IMAGES="./data/kielce_university_of_technology/train/images",
                                            VAL_IMAGES="./data/kielce_university_of_technology/valid/images",
                                            VAL_ANNOTATIONS="./data/kielce_university_of_technology/valid/xml",
                                            IMG_WIDTH=320,
                                            EPOCHS=150,
                                            BATCH_SIZE=7,
                                            num_classes=3,
                                            label_to_index={'car': 2, 'bus': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                            TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                            VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                            VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/xml",
                                            IMG_WIDTH=480,
                                            EPOCHS=50,
                                            BATCH_SIZE=7,
                                            num_classes=3,
                                            label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                            TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                            VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                            VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/xml",
                                            IMG_WIDTH=480,
                                            EPOCHS=100,
                                            BATCH_SIZE=7,
                                            num_classes=3,
                                            label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

retinanet_resnet50_fpn.retinanet_resnet50_fpn(TRAIN_ANNOTATIONS="./data/toggenburg_alpaca_ranch/train/xml",
                                            TRAIN_IMAGES="./data/toggenburg_alpaca_ranch/train/images",
                                            VAL_IMAGES="./data/toggenburg_alpaca_ranch/valid/images",
                                            VAL_ANNOTATIONS="./data/toggenburg_alpaca_ranch/valid/xml",
                                            IMG_WIDTH=480,
                                            EPOCHS=150,
                                            BATCH_SIZE=7,
                                            num_classes=3,
                                            label_to_index={'sheep': 2, 'car': 1, 'no_object': 0},
                                            PRE_TRAINED_BOOL=False)

model = YOLO('yolov8n.pt')
model.train(data='data/duomo/data.yaml', epochs=50, imgsz=640, batch=-1)

model = YOLO('yolov8n.pt')
model.train(data='data/duomo/data.yaml', epochs=100, imgsz=640, batch=-1)

model = YOLO('yolov8n.pt')
model.train(data='data/duomo/data.yaml', epochs=150, imgsz=640, batch=-1)

model = YOLO('yolov8n.pt')
model.train(data='data/hadji_dimitar_square/data.yaml', epochs=50, imgsz=320, batch=-1)

model = YOLO('yolov8n.pt')
model.train(data='data/hadji_dimitar_square/data.yaml', epochs=100, imgsz=320, batch=-1)

model = YOLO('yolov8n.pt')
model.train(data='data/hadji_dimitar_square/data.yaml', epochs=150, imgsz=320, batch=-1)

model = YOLO('yolov8n.pt')
model.train(data='data/keskvaljak/data.yaml', epochs=50, imgsz=320, batch=-1)

model = YOLO('yolov8n.pt')
model.train(data='data/keskvaljak/data.yaml', epochs=100, imgsz=320, batch=-1)

model = YOLO('yolov8n.pt')
model.train(data='data/keskvaljak/data.yaml', epochs=150, imgsz=320, batch=-1)

model = YOLO('yolov8n.pt')
model.train(data='data/kielce_university_of_technology/data.yaml', epochs=50, imgsz=320, batch=-1)

model = YOLO('yolov8n.pt')
model.train(data='data/kielce_university_of_technology/data.yaml', epochs=100, imgsz=320, batch=-1)

model = YOLO('yolov8n.pt')
model.train(data='data/kielce_university_of_technology/data.yaml', epochs=150, imgsz=320, batch=-1)

model = YOLO('yolov8n.pt')
model.train(data='data/toggenburg_alpaca_ranch/data.yaml', epochs=50, imgsz=480, batch=-1)

model = YOLO('yolov8n.pt')
model.train(data='data/toggenburg_alpaca_ranch/data.yaml', epochs=100, imgsz=480, batch=-1)

model = YOLO('yolov8n.pt')
model.train(data='data/toggenburg_alpaca_ranch/data.yaml', epochs=150, imgsz=480, batch=-1)

model = YOLO('yolov8n.yaml')
model.train(data='data/duomo/data.yaml', epochs=50, imgsz=640, batch=-1)

model = YOLO('yolov8n.yaml')
model.train(data='data/duomo/data.yaml', epochs=100, imgsz=640, batch=-1)

model = YOLO('yolov8n.yaml')
model.train(data='data/duomo/data.yaml', epochs=150, imgsz=640, batch=-1)

model = YOLO('yolov8n.yaml')
model.train(data='data/hadji_dimitar_square/data.yaml', epochs=50, imgsz=320, batch=-1)

model = YOLO('yolov8n.yaml')
model.train(data='data/hadji_dimitar_square/data.yaml', epochs=100, imgsz=320, batch=-1)

model = YOLO('yolov8n.yaml')
model.train(data='data/hadji_dimitar_square/data.yaml', epochs=150, imgsz=320, batch=-1)

model = YOLO('yolov8n.yaml')
model.train(data='data/keskvaljak/data.yaml', epochs=50, imgsz=320, batch=-1)

model = YOLO('yolov8n.yaml')
model.train(data='data/keskvaljak/data.yaml', epochs=100, imgsz=320, batch=-1)

model = YOLO('yolov8n.yaml')
model.train(data='data/keskvaljak/data.yaml', epochs=150, imgsz=320, batch=-1)

model = YOLO('yolov8n.yaml')
model.train(data='data/kielce_university_of_technology/data.yaml', epochs=50, imgsz=320, batch=-1)

model = YOLO('yolov8n.yaml')
model.train(data='data/kielce_university_of_technology/data.yaml', epochs=100, imgsz=320, batch=-1)

model = YOLO('yolov8n.yaml')
model.train(data='data/kielce_university_of_technology/data.yaml', epochs=150, imgsz=320, batch=-1)

model = YOLO('yolov8n.yaml')
model.train(data='data/toggenburg_alpaca_ranch/data.yaml', epochs=50, imgsz=480, batch=-1)

model = YOLO('yolov8n.yaml')
model.train(data='data/toggenburg_alpaca_ranch/data.yaml', epochs=100, imgsz=480, batch=-1)

model = YOLO('yolov8n.yaml')
model.train(data='data/toggenburg_alpaca_ranch/data.yaml', epochs=150, imgsz=480, batch=-1)

#NON-hyper parameter tuned ^^

model = YOLO('yolov8n.pt')
model.train(data='data/duomo/data.yaml', epochs=50, imgsz=640, batch=-1, 
lr0= 0.00836, 
lrf= 0.00789, 
momentum= 0.90282, 
weight_decay= 0.00053, 
warmup_epochs= 2.58718, 
warmup_momentum= 0.82969, 
box= 5.06508, 
cls= 0.56455, 
dfl= 1.20986, 
hsv_h= 0.01215, 
hsv_s= 0.72735, 
hsv_v= 0.50865, 
degrees= 0.0, 
translate= 0.06612, 
scale= 0.33003, 
shear= 0.0, 
perspective= 0.0, 
flipud= 0.0, 
fliplr= 0.48993, 
bgr= 0.0, 
mosaic= 0.97529, 
mixup= 0.0, 
copy_paste= 0.0)

model = YOLO('yolov8n.pt')
model.train(data='data/duomo/data.yaml', epochs=100, imgsz=640, batch=-1, 
lr0= 0.00836, 
lrf= 0.00789, 
momentum= 0.90282, 
weight_decay= 0.00053, 
warmup_epochs= 2.58718, 
warmup_momentum= 0.82969, 
box= 5.06508, 
cls= 0.56455, 
dfl= 1.20986, 
hsv_h= 0.01215, 
hsv_s= 0.72735, 
hsv_v= 0.50865, 
degrees= 0.0, 
translate= 0.06612, 
scale= 0.33003, 
shear= 0.0, 
perspective= 0.0, 
flipud= 0.0, 
fliplr= 0.48993, 
bgr= 0.0, 
mosaic= 0.97529, 
mixup= 0.0, 
copy_paste= 0.0)

model = YOLO('yolov8n.pt')
model.train(data='data/duomo/data.yaml', epochs=150, imgsz=640, batch=-1, 
lr0= 0.00836, 
lrf= 0.00789, 
momentum= 0.90282, 
weight_decay= 0.00053, 
warmup_epochs= 2.58718, 
warmup_momentum= 0.82969, 
box= 5.06508, 
cls= 0.56455, 
dfl= 1.20986, 
hsv_h= 0.01215, 
hsv_s= 0.72735, 
hsv_v= 0.50865, 
degrees= 0.0, 
translate= 0.06612, 
scale= 0.33003, 
shear= 0.0, 
perspective= 0.0, 
flipud= 0.0, 
fliplr= 0.48993, 
bgr= 0.0, 
mosaic= 0.97529, 
mixup= 0.0, 
copy_paste= 0.0)

model = YOLO('yolov8n.pt')
model.train(data='data/hadji_dimitar_square/data.yaml', epochs=50, imgsz=320, batch=-1,
lr0=0.00757, 
lrf=0.0099, 
momentum=0.82658, 
weight_decay=0.00022, 
warmup_epochs=1.59433, 
warmup_momentum=0.92886, 
box=2.5462, 
cls=0.71163, 
dfl=1.92563, 
hsv_h=0.01152, 
hsv_s=0.76192, 
hsv_v=0.6631, 
degrees=0.0, 
translate=0.05195, 
scale=0.33233, 
shear=0.0, 
perspective=0.0, 
flipud=0.0, 
fliplr=0.61564, 
bgr=0.0, 
mosaic=0.90256, 
mixup=0.0, 
copy_paste=0.0)

model = YOLO('yolov8n.pt')
model.train(data='data/hadji_dimitar_square/data.yaml', epochs=100, imgsz=320, batch=-1,
lr0=0.00757, 
lrf=0.0099, 
momentum=0.82658, 
weight_decay=0.00022, 
warmup_epochs=1.59433, 
warmup_momentum=0.92886, 
box=2.5462, 
cls=0.71163, 
dfl=1.92563, 
hsv_h=0.01152, 
hsv_s=0.76192, 
hsv_v=0.6631, 
degrees=0.0, 
translate=0.05195, 
scale=0.33233, 
shear=0.0, 
perspective=0.0, 
flipud=0.0, 
fliplr=0.61564, 
bgr=0.0, 
mosaic=0.90256, 
mixup=0.0, 
copy_paste=0.0)

model = YOLO('yolov8n.pt')
model.train(data='data/hadji_dimitar_square/data.yaml', epochs=150, imgsz=320, batch=-1,
lr0=0.00757, 
lrf=0.0099, 
momentum=0.82658, 
weight_decay=0.00022, 
warmup_epochs=1.59433, 
warmup_momentum=0.92886, 
box=2.5462, 
cls=0.71163, 
dfl=1.92563, 
hsv_h=0.01152, 
hsv_s=0.76192, 
hsv_v=0.6631, 
degrees=0.0, 
translate=0.05195, 
scale=0.33233, 
shear=0.0, 
perspective=0.0, 
flipud=0.0, 
fliplr=0.61564, 
bgr=0.0, 
mosaic=0.90256, 
mixup=0.0, 
copy_paste=0.0)

model = YOLO('yolov8n.pt')
model.train(data='data/keskvaljak/data.yaml', epochs=50, imgsz=320, batch=-1,
lr0=0.00782, 
lrf=0.00734, 
momentum=0.87459, 
weight_decay=0.00046, 
warmup_epochs=4.73128, 
warmup_momentum=0.67091, 
box=11.25887, 
cls=0.46844, 
dfl=2.0365, 
hsv_h=0.01085, 
hsv_s=0.75683, 
hsv_v=0.25776, 
degrees=0.0, 
translate=0.0753, 
scale=0.33083, 
shear=0.0, 
perspective=0.0, 
flipud=0.0, 
fliplr=0.7223, 
bgr=0.0, 
mosaic=0.90357, 
mixup=0.0, 
copy_paste=0.0)

model = YOLO('yolov8n.pt')
model.train(data='data/keskvaljak/data.yaml', epochs=100, imgsz=320, batch=-1,
lr0=0.00782, 
lrf=0.00734, 
momentum=0.87459, 
weight_decay=0.00046, 
warmup_epochs=4.73128, 
warmup_momentum=0.67091, 
box=11.25887, 
cls=0.46844, 
dfl=2.0365, 
hsv_h=0.01085, 
hsv_s=0.75683, 
hsv_v=0.25776, 
degrees=0.0, 
translate=0.0753, 
scale=0.33083, 
shear=0.0, 
perspective=0.0, 
flipud=0.0, 
fliplr=0.7223, 
bgr=0.0, 
mosaic=0.90357, 
mixup=0.0, 
copy_paste=0.0)

model = YOLO('yolov8n.pt')
model.train(data='data/keskvaljak/data.yaml', epochs=150, imgsz=320, batch=-1,
lr0=0.00782, 
lrf=0.00734, 
momentum=0.87459, 
weight_decay=0.00046, 
warmup_epochs=4.73128, 
warmup_momentum=0.67091, 
box=11.25887, 
cls=0.46844, 
dfl=2.0365, 
hsv_h=0.01085, 
hsv_s=0.75683, 
hsv_v=0.25776, 
degrees=0.0, 
translate=0.0753, 
scale=0.33083, 
shear=0.0, 
perspective=0.0, 
flipud=0.0, 
fliplr=0.7223, 
bgr=0.0, 
mosaic=0.90357, 
mixup=0.0, 
copy_paste=0.0)

model = YOLO('yolov8n.pt')
model.train(data='data/kielce_university_of_technology/data.yaml', epochs=50, imgsz=320, batch=-1,
lr0=0.00967, 
lrf=0.01187, 
momentum=0.78674, 
weight_decay=0.00021, 
warmup_epochs=3.03666, 
warmup_momentum=0.81758, 
box=5.25473, 
cls=0.35605, 
dfl=1.53431, 
hsv_h=0.01101, 
hsv_s=0.51982, 
hsv_v=0.49718, 
degrees=0.0, 
translate=0.06107, 
scale=0.37808, 
shear=0.0, 
perspective=0.0, 
flipud=0.0, 
fliplr=0.43225, 
bgr=0.0, 
mosaic=0.88523, 
mixup=0.0, 
copy_paste=0.0)

model = YOLO('yolov8n.pt')
model.train(data='data/kielce_university_of_technology/data.yaml', epochs=100, imgsz=320, batch=-1,
lr0=0.00967, 
lrf=0.01187, 
momentum=0.78674, 
weight_decay=0.00021, 
warmup_epochs=3.03666, 
warmup_momentum=0.81758, 
box=5.25473, 
cls=0.35605, 
dfl=1.53431, 
hsv_h=0.01101, 
hsv_s=0.51982, 
hsv_v=0.49718, 
degrees=0.0, 
translate=0.06107, 
scale=0.37808, 
shear=0.0, 
perspective=0.0, 
flipud=0.0, 
fliplr=0.43225, 
bgr=0.0, 
mosaic=0.88523, 
mixup=0.0, 
copy_paste=0.0)

model = YOLO('yolov8n.pt')
model.train(data='data/kielce_university_of_technology/data.yaml', epochs=150, imgsz=320, batch=-1,
lr0=0.00967, 
lrf=0.01187, 
momentum=0.78674, 
weight_decay=0.00021, 
warmup_epochs=3.03666, 
warmup_momentum=0.81758, 
box=5.25473, 
cls=0.35605, 
dfl=1.53431, 
hsv_h=0.01101, 
hsv_s=0.51982, 
hsv_v=0.49718, 
degrees=0.0, 
translate=0.06107, 
scale=0.37808, 
shear=0.0, 
perspective=0.0, 
flipud=0.0, 
fliplr=0.43225, 
bgr=0.0, 
mosaic=0.88523, 
mixup=0.0, 
copy_paste=0.0)

model = YOLO('yolov8n.pt')
model.train(data='data/toggenburg_alpaca_ranch/data.yaml', epochs=50, imgsz=480, batch=-1,
lr0=0.01138, 
lrf=0.01039, 
momentum=0.87794, 
weight_decay=0.00053, 
warmup_epochs=3.49812, 
warmup_momentum=0.78998, 
box=7.87804, 
cls=0.52783, 
dfl=1.028, 
hsv_h=0.01762, 
hsv_s=0.603, 
hsv_v=0.48745, 
degrees=0.0, 
translate=0.07056, 
scale=0.31698, 
shear=0.0, 
perspective=0.0, 
flipud=0.0, 
fliplr=0.38476, 
bgr=0.0, 
mosaic=0.91511, 
mixup=0.0, 
copy_paste=0.0)

model = YOLO('yolov8n.pt')
model.train(data='data/toggenburg_alpaca_ranch/data.yaml', epochs=100, imgsz=480, batch=-1,
lr0=0.01138, 
lrf=0.01039, 
momentum=0.87794, 
weight_decay=0.00053, 
warmup_epochs=3.49812, 
warmup_momentum=0.78998, 
box=7.87804, 
cls=0.52783, 
dfl=1.028, 
hsv_h=0.01762, 
hsv_s=0.603, 
hsv_v=0.48745, 
degrees=0.0, 
translate=0.07056, 
scale=0.31698, 
shear=0.0, 
perspective=0.0, 
flipud=0.0, 
fliplr=0.38476, 
bgr=0.0, 
mosaic=0.91511, 
mixup=0.0, 
copy_paste=0.0)

model = YOLO('yolov8n.pt')
model.train(data='data/toggenburg_alpaca_ranch/data.yaml', epochs=150, imgsz=480, batch=-1,
lr0=0.01138, 
lrf=0.01039, 
momentum=0.87794, 
weight_decay=0.00053, 
warmup_epochs=3.49812, 
warmup_momentum=0.78998, 
box=7.87804, 
cls=0.52783, 
dfl=1.028, 
hsv_h=0.01762, 
hsv_s=0.603, 
hsv_v=0.48745, 
degrees=0.0, 
translate=0.07056, 
scale=0.31698, 
shear=0.0, 
perspective=0.0, 
flipud=0.0, 
fliplr=0.38476, 
bgr=0.0, 
mosaic=0.91511, 
mixup=0.0, 
copy_paste=0.0)

## marker

model = YOLO('yolov8n.yaml')
model.train(data='data/duomo/data.yaml', epochs=50, imgsz=640, batch=-1, 
lr0= 0.00836, 
lrf= 0.00789, 
momentum= 0.90282, 
weight_decay= 0.00053, 
warmup_epochs= 2.58718, 
warmup_momentum= 0.82969, 
box= 5.06508, 
cls= 0.56455, 
dfl= 1.20986, 
hsv_h= 0.01215, 
hsv_s= 0.72735, 
hsv_v= 0.50865, 
degrees= 0.0, 
translate= 0.06612, 
scale= 0.33003, 
shear= 0.0, 
perspective= 0.0, 
flipud= 0.0, 
fliplr= 0.48993, 
bgr= 0.0, 
mosaic= 0.97529, 
mixup= 0.0, 
copy_paste= 0.0)

model = YOLO('yolov8n.yaml')
model.train(data='data/duomo/data.yaml', epochs=100, imgsz=640, batch=-1, 
lr0= 0.00836, 
lrf= 0.00789, 
momentum= 0.90282, 
weight_decay= 0.00053, 
warmup_epochs= 2.58718, 
warmup_momentum= 0.82969, 
box= 5.06508, 
cls= 0.56455, 
dfl= 1.20986, 
hsv_h= 0.01215, 
hsv_s= 0.72735, 
hsv_v= 0.50865, 
degrees= 0.0, 
translate= 0.06612, 
scale= 0.33003, 
shear= 0.0, 
perspective= 0.0, 
flipud= 0.0, 
fliplr= 0.48993, 
bgr= 0.0, 
mosaic= 0.97529, 
mixup= 0.0, 
copy_paste= 0.0)

model = YOLO('yolov8n.yaml')
model.train(data='data/duomo/data.yaml', epochs=150, imgsz=640, batch=-1, 
lr0= 0.00836, 
lrf= 0.00789, 
momentum= 0.90282, 
weight_decay= 0.00053, 
warmup_epochs= 2.58718, 
warmup_momentum= 0.82969, 
box= 5.06508, 
cls= 0.56455, 
dfl= 1.20986, 
hsv_h= 0.01215, 
hsv_s= 0.72735, 
hsv_v= 0.50865, 
degrees= 0.0, 
translate= 0.06612, 
scale= 0.33003, 
shear= 0.0, 
perspective= 0.0, 
flipud= 0.0, 
fliplr= 0.48993, 
bgr= 0.0, 
mosaic= 0.97529, 
mixup= 0.0, 
copy_paste= 0.0)

model = YOLO('yolov8n.yaml')
model.train(data='data/hadji_dimitar_square/data.yaml', epochs=50, imgsz=320, batch=-1,
lr0=0.00757, 
lrf=0.0099, 
momentum=0.82658, 
weight_decay=0.00022, 
warmup_epochs=1.59433, 
warmup_momentum=0.92886, 
box=2.5462, 
cls=0.71163, 
dfl=1.92563, 
hsv_h=0.01152, 
hsv_s=0.76192, 
hsv_v=0.6631, 
degrees=0.0, 
translate=0.05195, 
scale=0.33233, 
shear=0.0, 
perspective=0.0, 
flipud=0.0, 
fliplr=0.61564, 
bgr=0.0, 
mosaic=0.90256, 
mixup=0.0, 
copy_paste=0.0)

model = YOLO('yolov8n.yaml')
model.train(data='data/hadji_dimitar_square/data.yaml', epochs=100, imgsz=320, batch=-1,
lr0=0.00757, 
lrf=0.0099, 
momentum=0.82658, 
weight_decay=0.00022, 
warmup_epochs=1.59433, 
warmup_momentum=0.92886, 
box=2.5462, 
cls=0.71163, 
dfl=1.92563, 
hsv_h=0.01152, 
hsv_s=0.76192, 
hsv_v=0.6631, 
degrees=0.0, 
translate=0.05195, 
scale=0.33233, 
shear=0.0, 
perspective=0.0, 
flipud=0.0, 
fliplr=0.61564, 
bgr=0.0, 
mosaic=0.90256, 
mixup=0.0, 
copy_paste=0.0)

model = YOLO('yolov8n.yaml')
model.train(data='data/hadji_dimitar_square/data.yaml', epochs=150, imgsz=320, batch=-1,
lr0=0.00757, 
lrf=0.0099, 
momentum=0.82658, 
weight_decay=0.00022, 
warmup_epochs=1.59433, 
warmup_momentum=0.92886, 
box=2.5462, 
cls=0.71163, 
dfl=1.92563, 
hsv_h=0.01152, 
hsv_s=0.76192, 
hsv_v=0.6631, 
degrees=0.0, 
translate=0.05195, 
scale=0.33233, 
shear=0.0, 
perspective=0.0, 
flipud=0.0, 
fliplr=0.61564, 
bgr=0.0, 
mosaic=0.90256, 
mixup=0.0, 
copy_paste=0.0)

model = YOLO('yolov8n.yaml')
model.train(data='data/keskvaljak/data.yaml', epochs=50, imgsz=320, batch=-1,
lr0=0.00782, 
lrf=0.00734, 
momentum=0.87459, 
weight_decay=0.00046, 
warmup_epochs=4.73128, 
warmup_momentum=0.67091, 
box=11.25887, 
cls=0.46844, 
dfl=2.0365, 
hsv_h=0.01085, 
hsv_s=0.75683, 
hsv_v=0.25776, 
degrees=0.0, 
translate=0.0753, 
scale=0.33083, 
shear=0.0, 
perspective=0.0, 
flipud=0.0, 
fliplr=0.7223, 
bgr=0.0, 
mosaic=0.90357, 
mixup=0.0, 
copy_paste=0.0)

model = YOLO('yolov8n.yaml')
model.train(data='data/keskvaljak/data.yaml', epochs=100, imgsz=320, batch=-1,
lr0=0.00782, 
lrf=0.00734, 
momentum=0.87459, 
weight_decay=0.00046, 
warmup_epochs=4.73128, 
warmup_momentum=0.67091, 
box=11.25887, 
cls=0.46844, 
dfl=2.0365, 
hsv_h=0.01085, 
hsv_s=0.75683, 
hsv_v=0.25776, 
degrees=0.0, 
translate=0.0753, 
scale=0.33083, 
shear=0.0, 
perspective=0.0, 
flipud=0.0, 
fliplr=0.7223, 
bgr=0.0, 
mosaic=0.90357, 
mixup=0.0, 
copy_paste=0.0)

model = YOLO('yolov8n.yaml')
model.train(data='data/keskvaljak/data.yaml', epochs=150, imgsz=320, batch=-1,
lr0=0.00782, 
lrf=0.00734, 
momentum=0.87459, 
weight_decay=0.00046, 
warmup_epochs=4.73128, 
warmup_momentum=0.67091, 
box=11.25887, 
cls=0.46844, 
dfl=2.0365, 
hsv_h=0.01085, 
hsv_s=0.75683, 
hsv_v=0.25776, 
degrees=0.0, 
translate=0.0753, 
scale=0.33083, 
shear=0.0, 
perspective=0.0, 
flipud=0.0, 
fliplr=0.7223, 
bgr=0.0, 
mosaic=0.90357, 
mixup=0.0, 
copy_paste=0.0)

model = YOLO('yolov8n.yaml')
model.train(data='data/kielce_university_of_technology/data.yaml', epochs=50, imgsz=320, batch=-1,
lr0=0.00967, 
lrf=0.01187, 
momentum=0.78674, 
weight_decay=0.00021, 
warmup_epochs=3.03666, 
warmup_momentum=0.81758, 
box=5.25473, 
cls=0.35605, 
dfl=1.53431, 
hsv_h=0.01101, 
hsv_s=0.51982, 
hsv_v=0.49718, 
degrees=0.0, 
translate=0.06107, 
scale=0.37808, 
shear=0.0, 
perspective=0.0, 
flipud=0.0, 
fliplr=0.43225, 
bgr=0.0, 
mosaic=0.88523, 
mixup=0.0, 
copy_paste=0.0)

model = YOLO('yolov8n.yaml')
model.train(data='data/kielce_university_of_technology/data.yaml', epochs=100, imgsz=320, batch=-1,
lr0=0.00967, 
lrf=0.01187, 
momentum=0.78674, 
weight_decay=0.00021, 
warmup_epochs=3.03666, 
warmup_momentum=0.81758, 
box=5.25473, 
cls=0.35605, 
dfl=1.53431, 
hsv_h=0.01101, 
hsv_s=0.51982, 
hsv_v=0.49718, 
degrees=0.0, 
translate=0.06107, 
scale=0.37808, 
shear=0.0, 
perspective=0.0, 
flipud=0.0, 
fliplr=0.43225, 
bgr=0.0, 
mosaic=0.88523, 
mixup=0.0, 
copy_paste=0.0)

model = YOLO('yolov8n.yaml')
model.train(data='data/kielce_university_of_technology/data.yaml', epochs=150, imgsz=320, batch=-1,
lr0=0.00967, 
lrf=0.01187, 
momentum=0.78674, 
weight_decay=0.00021, 
warmup_epochs=3.03666, 
warmup_momentum=0.81758, 
box=5.25473, 
cls=0.35605, 
dfl=1.53431, 
hsv_h=0.01101, 
hsv_s=0.51982, 
hsv_v=0.49718, 
degrees=0.0, 
translate=0.06107, 
scale=0.37808, 
shear=0.0, 
perspective=0.0, 
flipud=0.0, 
fliplr=0.43225, 
bgr=0.0, 
mosaic=0.88523, 
mixup=0.0, 
copy_paste=0.0)

model = YOLO('yolov8n.yaml')
model.train(data='data/toggenburg_alpaca_ranch/data.yaml', epochs=50, imgsz=480, batch=-1,
lr0=0.01138, 
lrf=0.01039, 
momentum=0.87794, 
weight_decay=0.00053, 
warmup_epochs=3.49812, 
warmup_momentum=0.78998, 
box=7.87804, 
cls=0.52783, 
dfl=1.028, 
hsv_h=0.01762, 
hsv_s=0.603, 
hsv_v=0.48745, 
degrees=0.0, 
translate=0.07056, 
scale=0.31698, 
shear=0.0, 
perspective=0.0, 
flipud=0.0, 
fliplr=0.38476, 
bgr=0.0, 
mosaic=0.91511, 
mixup=0.0, 
copy_paste=0.0)

model = YOLO('yolov8n.yaml')
model.train(data='data/toggenburg_alpaca_ranch/data.yaml', epochs=100, imgsz=480, batch=-1,
lr0=0.01138, 
lrf=0.01039, 
momentum=0.87794, 
weight_decay=0.00053, 
warmup_epochs=3.49812, 
warmup_momentum=0.78998, 
box=7.87804, 
cls=0.52783, 
dfl=1.028, 
hsv_h=0.01762, 
hsv_s=0.603, 
hsv_v=0.48745, 
degrees=0.0, 
translate=0.07056, 
scale=0.31698, 
shear=0.0, 
perspective=0.0, 
flipud=0.0, 
fliplr=0.38476, 
bgr=0.0, 
mosaic=0.91511, 
mixup=0.0, 
copy_paste=0.0)

model = YOLO('yolov8n.yaml')
model.train(data='data/toggenburg_alpaca_ranch/data.yaml', epochs=150, imgsz=480, batch=-1,
lr0=0.01138, 
lrf=0.01039, 
momentum=0.87794, 
weight_decay=0.00053, 
warmup_epochs=3.49812, 
warmup_momentum=0.78998, 
box=7.87804, 
cls=0.52783, 
dfl=1.028, 
hsv_h=0.01762, 
hsv_s=0.603, 
hsv_v=0.48745, 
degrees=0.0, 
translate=0.07056, 
scale=0.31698, 
shear=0.0, 
perspective=0.0, 
flipud=0.0, 
fliplr=0.38476, 
bgr=0.0, 
mosaic=0.91511, 
mixup=0.0, 
copy_paste=0.0)

## Hyper parameter tuned ^^

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
