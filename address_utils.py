import re
import requests

class AddressFormatter:
    def __init__(self):
        # 常见省/州名称映射（可根据需要扩展）
        self.state_mapping = {
            # 葡萄牙语州名映射
            'são paulo': 'SP', 'sao paulo': 'SP', 'sp': 'SP',
            'rio de janeiro': 'RJ', 'rj': 'RJ',
            'minas gerais': 'MG', 'mg': 'MG',
            # 英语州名映射
            'california': 'CA', 'ca': 'CA',
            'new york': 'NY', 'ny': 'NY',
            'texas': 'TX', 'tx': 'TX'
        }
        
        # 国家代码映射
        self.country_mapping = {
            'brasil': 'BR', 'brazil': 'BR',
            'estados unidos': 'US', 'united states': 'US', 'usa': 'US'
        }

    def clean_address(self, address):
        """
        第一步：清理地址，去除多余空格和特殊字符
        """
        if not address:
            return ""
            
        # 转换为小写并去除首尾空格
        cleaned = address.lower().strip()
        
        # 替换多个连续空格为单个空格
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # 处理常见的分隔符问题
        cleaned = re.sub(r'[,\-\.]+', ', ', cleaned)
        
        # 清理多余的空格和逗号组合
        cleaned = re.sub(r',\s*,', ',', cleaned)
        cleaned = re.sub(r'\s*,\s*', ', ', cleaned)
        
        # 去除特殊字符，只保留字母、数字、空格和基本标点
        cleaned = re.sub(r'[^\w\s,\.\-]', '', cleaned)
        
        # 再次清理空格
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned

    def validate_zip_code(self, zip_code, country='BR'):
        """
        验证和标准化邮政编码
        """
        if not zip_code:
            return None
            
        # 移除非数字字符
        clean_zip = re.sub(r'\D', '', zip_code)
        
        if country.upper() == 'BR':  # 巴西邮政编码
            if len(clean_zip) == 8:
                return f"{clean_zip[:5]}-{clean_zip[5:]}"
            elif len(clean_zip) == 5:
                return clean_zip
                
        elif country.upper() == 'US':  # 美国邮政编码
            if len(clean_zip) == 5:
                return clean_zip
            elif len(clean_zip) == 9:
                return f"{clean_zip[:5]}-{clean_zip[5:]}"
                
        return None

    def standardize_state(self, state_name):
        """
        标准化省/州名称
        """
        if not state_name:
            return None
            
        clean_state = state_name.lower().strip()
        return self.state_mapping.get(clean_state, state_name.upper())

    def extract_zip_code(self, address):
        """
        从地址中提取邮政编码
        """
        # 巴西邮政编码模式：XXXXX-XXX 或 XXXXX
        br_zip_pattern = r'\b(\d{5}-\d{3}|\d{5})\b'
        
        # 美国邮政编码模式：XXXXX 或 XXXXX-XXXX
        us_zip_pattern = r'\b(\d{5}-\d{4}|\d{5})\b'
        
        # 尝试匹配巴西邮政编码
        br_match = re.search(br_zip_pattern, address)
        if br_match:
            return br_match.group(), 'BR'
            
        # 尝试匹配美国邮政编码
        us_match = re.search(us_zip_pattern, address)
        if us_match:
            return us_match.group(), 'US'
            
        return None, None

    def parse_address_components(self, address):
        """
        解析地址组件（简单版本）
        """
        components = {
            'street': '',
            'city': '',
            'state': '',
            'zip_code': '',
            'country': ''
        }
        
        # 按逗号分割地址组件
        parts = [part.strip() for part in address.split(',')]
        
        if len(parts) >= 3:
            components['street'] = parts[0]
            components['city'] = parts[1]
            
            # 最后一个部分可能包含州和邮政编码
            last_part = parts[-1]
            state_zip_match = re.search(r'([A-Za-z\s]+)\s*(\d.*)?', last_part)
            if state_zip_match:
                components['state'] = state_zip_match.group(1).strip()
                if state_zip_match.group(2):
                    components['zip_code'] = state_zip_match.group(2).strip()
        
        return components

    def format_address(self, address):
        """
        主函数：格式化并验证地址
        """
        if not address:
            return None
            
        try:
            # 第一步：清理地址
            cleaned_address = self.clean_address(address)
            print(f"清理后地址: {cleaned_address}")
            
            # 提取邮政编码和国家
            zip_code, country = self.extract_zip_code(cleaned_address)
            
            # 验证和标准化邮政编码
            standardized_zip = None
            if zip_code:
                standardized_zip = self.validate_zip_code(zip_code, country)
                print(f"标准化邮政编码: {standardized_zip}")
            
            # 解析地址组件
            components = self.parse_address_components(cleaned_address)
            
            # 标准化州名
            if components['state']:
                components['state'] = self.standardize_state(components['state'])
                print(f"标准化州名: {components['state']}")
            
            # 构建标准化地址
            standardized_parts = []
            if components['street']:
                standardized_parts.append(components['street'])
            if components['city']:
                standardized_parts.append(components['city'])
            if components['state']:
                standardized_parts.append(components['state'])
            if standardized_zip:
                standardized_parts.append(standardized_zip)
            if country:
                standardized_parts.append(country)
            
            standardized_address = ', '.join(standardized_parts)
            
            return {
                'original_address': address,
                'cleaned_address': cleaned_address,
                'standardized_address': standardized_address,
                'components': components,
                'zip_code': standardized_zip,
                'country': country
            }
            
        except Exception as e:
            print(f"地址处理错误: {e}")
            return None

# 使用示例
def main():
    formatter = AddressFormatter()
    
    # 测试地址示例
    test_addresses = [
        # 葡萄牙语地址
        "Rua das Flores,, 123  -  Centro,, São Paulo  ,, SP , 01234-567",
        "Av. Paulista, 1000,, Bela Vista, São Paulo, SP, 01310-100",
        
        # 英语地址
        "123 Main St ,,  Springfield , IL , 62704- 1234",
        "456 Oak Avenue,,, Los Angeles,, CA, 90001"
    ]
    
    for address in test_addresses:
        print(f"\n原始地址: {address}")
        result = formatter.format_address(address)
        
        if result:
            print(f"标准化地址: {result['standardized_address']}")
            print(f"地址组件: {result['components']}")
        print("-" * 50)

if __name__ == "__main__":
    main()