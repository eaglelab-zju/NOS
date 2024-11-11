def get_screen_configs(dataset_config, page_id):
    if "mixed" in dataset_config.name:
        if page_id >= 0:
            phone_height, phone_width, extra_bottom, length_threshold = (
                dataset_config.a11y_phone_height,
                dataset_config.a11y_phone_width,
                dataset_config.a11y_extra_bottom,
                dataset_config.a11y_length_threshold,
            )

        else:
            phone_height, phone_width, extra_bottom, length_threshold = (
                dataset_config.rico_phone_height,
                dataset_config.rico_phone_width,
                dataset_config.rico_extra_bottom,
                dataset_config.rico_length_threshold,
            )

    else:
        phone_height, phone_width, extra_bottom, length_threshold = (
            dataset_config.phone_height,
            dataset_config.phone_width,
            dataset_config.extra_bottom,
            dataset_config.length_threshold,
        )
    return phone_height, phone_width, extra_bottom, length_threshold


def is_valid(
    node,
    phone_height,
    phone_width,
    extra_bottom,
    length_threshold=5,
    inplace=False,
):
    # 指定框的坐标和大小
    left, right, top, bottom = (
        node["screen_left"],
        node["screen_right"],
        node["screen_top"],
        node["screen_bottom"],
    )

    # 完全超出上面
    if bottom <= 0:
        return False
    # 完全超出底部
    if top >= phone_height - extra_bottom:
        return False
    # 完全超出左边
    if right <= 0:
        return False
    # 完全超出右边
    if left >= phone_width:
        return False

    # 修正溢出框
    top = max(top, 0)
    bottom = min(bottom, phone_height - extra_bottom)
    left = max(left, 0)
    right = min(right, phone_width)

    if inplace:
        (
            node["screen_left"],
            node["screen_right"],
            node["screen_top"],
            node["screen_bottom"],
        ) = (left, right, top, bottom)
        
    if 'is_valid' in node:
        return node['is_valid']
    return (
        left + length_threshold < right
        and top + length_threshold < bottom
    )


def is_focusable(node):
    return node["focusable_manual_label"]


def get_node_weight(config, page_id, node):
    weight = 1
    if node["clickable"] != is_focusable(node):
        weight = config.dataset_config.clickable_weight
    elif page_id >= config.dataset_config.labelled_data_id_threshold:
        weight = config.dataset_config.manual_weight
    return weight


def get_coordinators(node, phone_height, phone_width, extra_bottom, length_threshold=5):
    is_valid(
        node,
        phone_height,
        phone_width,
        extra_bottom,
        length_threshold,
        inplace=True,
    )
    return [
        node["screen_left"],
        node["screen_right"],
        node["screen_top"],
        node["screen_bottom"],
    ]


def graph2adj(g):
    return g.adj_external(scipy_fmt="csr").todense()


def graph2edgeindex(g):
    return g.adj_external(scipy_fmt="csr").nonzero()
