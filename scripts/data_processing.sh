#!/usr/bin/env bash
echo "Data_processing"
echo "Data_processing 10% RefCOCO"
python pick_part_dataset.py --dataset=refcoco --save-dir=anns/refcoco/ --pick-percent=0.1
echo "Data_processing 5% RefCOCO"
python pick_part_dataset.py --dataset=refcoco --save-dir=anns/refcoco/ --pick-percent=0.05
echo "Data_processing 1% RefCOCO"
python pick_part_dataset.py --dataset=refcoco --save-dir=anns/refcoco/ --pick-percent=0.01
echo "Data_processing 0.1% RefCOCO"
python pick_part_dataset.py --dataset=refcoco --save-dir=anns/refcoco/ --pick-percent=0.001

echo "Data_processing 10% RefCOCO+"
python pick_part_dataset.py --dataset=refcoco+ --save-dir=anns/refcoco+/ --pick-percent=0.1
echo "Data_processing 5% RefCOCO+"
python pick_part_dataset.py --dataset=refcoco+ --save-dir=anns/refcoco+/ --pick-percent=0.05
echo "Data_processing 1% RefCOCO+"
python pick_part_dataset.py --dataset=refcoco+ --save-dir=anns/refcoco+/ --pick-percent=0.01
echo "Data_processing 0.1% RefCOCO+"
python pick_part_dataset.py --dataset=refcoco+ --save-dir=anns/refcoco+/ --pick-percent=0.001

echo "Data_processing 10% RefCOCOg"
python pick_part_dataset.py --dataset=refcocog --save-dir=anns/refcocog/ --pick-percent=0.1
echo "Data_processing 5% RefCOCOg"
python pick_part_dataset.py --dataset=refcocog --save-dir=anns/refcocog/ --pick-percent=0.05
echo "Data_processing 1% RefCOCOg"
python pick_part_dataset.py --dataset=refcocog --save-dir=anns/refcocog/ --pick-percent=0.01
echo "Data_processing 0.1% RefCOCOg"
python pick_part_dataset.py --dataset=refcocog --save-dir=anns/refcocog/ --pick-percent=0.001

echo "end"