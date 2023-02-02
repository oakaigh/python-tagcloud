# TODO 
def _SummedAreaTable_rm_area_matrix_comp(self, block_size: Dimension):
    x_max, y_max = self.base.shape
    x_block, y_block = block_size

    res = np.full(((x_max - x_block), (y_max - y_block)), fill_value=np.nan)
    for x in range(x_max - x_block):
        for y in range(y_max - y_block):
            offset = Coordinate(x, y)
            res[x, y] = self.area(
                offset=offset, 
                block_size=block_size
            )

    return res