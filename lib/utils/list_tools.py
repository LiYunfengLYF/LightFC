def split_list(obj_lisat, node=4):
    num = int(len(obj_lisat) / node) + 1
    sublist = []
    for i in range(node):
        sublist.append(obj_lisat[i * num:(i + 1) * num])
    return sublist
