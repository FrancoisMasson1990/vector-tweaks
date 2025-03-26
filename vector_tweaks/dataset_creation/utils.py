from typing import Any


class ProductParserUtils:
    def from_string(value: Any) -> str:
        return " " + str(value) if value is not None else ""

    def from_raw_currency(value: Any) -> str:
        return " " + f"{value['value']} {value['currency']}" if value is not None else ""

    def from_list(value: Any) -> str:
        return "\n- " + "\n- ".join(value) if value is not None else ""

    def from_list_head(value: Any) -> str:
        return " " + str(value[0]) if value is not None else ""

    # map field name from json representation to a human (llm) readable format
    # each field comes with a mapper function that converts the field value from the json to a string representation
    core_field_metadata: dict[str, dict[str, Any]] = {
        "title": {"pretty_name": "Product name", "to_str": from_string},
        "link": {"pretty_name": "Product url", "to_str": from_string},
        "id": {"pretty_name": "Product reference", "to_str": from_string},
        "itemGroupId": {"pretty_name": "Product group reference", "to_str": from_string},
        "imageLink": {"pretty_name": "Image link", "to_str": from_string},
        "additionalImageLink": {"pretty_name": "Additional image link", "to_str": from_string},
        "productTypes": {"pretty_name": "Product types", "to_str": from_list},
        "availability": {"pretty_name": "Availability", "to_str": from_string},
        "availabilityDate": {"pretty_name": "Availability date", "to_str": from_string},
        "price": {"pretty_name": "Price", "to_str": from_raw_currency},
        "salePrice": {"pretty_name": "Sale price", "to_str": from_raw_currency},
        "brand": {"pretty_name": "Brand name", "to_str": from_string},
        "description": {"pretty_name": "Product description", "to_str": from_string},
        "color": {"pretty_name": "Product color", "to_str": from_string},
        "sizes": {"pretty_name": "Product size", "to_str": from_list_head},
        "condition": {"pretty_name": "Condition", "to_str": from_string},
        "gender": {"pretty_name": "Gender", "to_str": from_string},
        "material": {"pretty_name": "Material", "to_str": from_string},
    }

    @staticmethod
    def pretty_print_product_format(product_dict: dict[str, Any], excluded_fields: list[str] | None = None) -> str:
        """
           Pretty print a product dictionary, making it more readable for a llm
        :param product_dict: a json-like record that represents a product
        :param excluded_fields: an optional list of fields that will not appear in the pretty print
        :return: a string that represents the product in a more readable format for a llm
        """
        if not isinstance(product_dict, dict):
            raise ValueError("The product_dict parameter must be a dictionary")
        if excluded_fields is None:
            excluded_fields = []
        prompt = ""

        for field_name in ProductParserUtils.core_field_metadata:
            if field_name not in excluded_fields and field_name in product_dict:
                field_metadata = ProductParserUtils.core_field_metadata[field_name]
                field_pretty_name = field_metadata["pretty_name"]
                field_as_str = field_metadata["to_str"](product_dict[field_name])
                prompt += f"{field_pretty_name}:{field_as_str}\n"

        if "productDetails" in product_dict and "productDetails" not in excluded_fields:
            prompt += "More product properties\n"
            for product_detail in product_dict["productDetails"]:
                property_name = product_detail["attributeName"]
                property_value = product_detail["attributeValue"]
                prompt += f"- {property_name}: {property_value}\n"
        return prompt

    @staticmethod
    def get_product_variants(product_dict: dict[str, Any]) -> list[str]:
        product_variants = []

        if "variants" in product_dict:
            variants_dicts = product_dict["variants"]
            if variants_dicts:
                for variant_dict in variants_dicts:
                    product_variants.append(ProductParserUtils.pretty_print_product_format(variant_dict))

        return product_variants
