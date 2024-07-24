import numpy as np
import os

annotation_path = 'cls_CT.txt'
with open(annotation_path, 'r') as f:
    # lines = f.readlines()
    lines = [line.strip() for line in f.readlines()]
np.random.seed(10101)
np.random.shuffle(lines)
np.random.seed(None)
num_valtest = int(len(lines) * 0.2)
num_test = int(num_valtest * 0.5)
num_val = int(num_valtest * 0.5)
num_train = len(lines) - num_valtest

lines_test = lines[:num_test]
lines_val = lines[num_test:num_valtest]
lines_train = lines[num_valtest:]

data_dic ={
    'num_test': lines_test,
    'num_val' : lines_val,
    'num_train': lines_train

}



if __name__ == '__main__':
    for se in 'num_train', 'num_val', 'num_test':
        list_file = open('cls_' + se + '.txt', 'w')
        for CT_Path in data_dic[se]:
            cls_id, CT_path = CT_Path.split(';')
            photo_names = os.listdir(CT_path)
            for photo_name in photo_names:
                list_file.write(str(cls_id) + ';' + '%s' % os.path.join(CT_path, photo_name))
                list_file.write('\n')
        list_file.close()
'''
/***
 *      ┌─┐       ┌─┐ + +
 *   ┌──┘ ┴───────┘ ┴──┐++
 *   │                 │
 *   │       ───       │++ + + +
 *   ███████───███████ │+
 *   │                 │+
 *   │       ─┴─       │
 *   │                 │
 *   └───┐         ┌───┘
 *       │         │
 *       │         │   + +
 *       │         │
 *       │         └──────────────┐
 *       │                        │
 *       │                        ├─┐
 *       │                        ┌─┘
 *       │                        │
 *       └─┐  ┐  ┌───────┬──┐  ┌──┘  + + + +
 *         │ ─┤ ─┤       │ ─┤ ─┤
 *         └──┴──┘       └──┴──┘  + + + +
 *                神兽保佑
 *               代码无BUG!
 */
'''