def res_scaler(input_res: list[int], scale: float) -> list[int]:
    """
    Scale resolution by a factor.
    
    Args:
        input_res: [height, width]
        scale: scaling factor
    
    Returns:
        scaled resolution [new_height, new_width]
    """
    return [int(input_res[0] * scale), int(input_res[1] * scale)]


