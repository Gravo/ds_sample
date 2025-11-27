import re
import pandas as pd
from typing import List, Dict, Optional

class PortugueseAddressFormatter:
    def __init__(self):
        # 巴西州名映射
        self.state_mapping = {
            'ac': 'AC', 'acre': 'AC',
            'al': 'AL', 'alagoas': 'AL',
            'ap': 'AP', 'amapá': 'AP', 'amapa': 'AP',
            'am': 'AM', 'amazonas': 'AM',
            'ba': 'BA', 'bahia': 'BA',
            'ce': 'CE', 'ceará': 'CE', 'ceara': 'CE',
            'df': 'DF', 'distrito federal': 'DF',
            'es': 'ES', 'espírito santo': 'ES', 'espirito santo': 'ES',
            'go': 'GO', 'goiás': 'GO', 'goias': 'GO',
            'ma': 'MA', 'maranhão': 'MA', 'maranhao': 'MA',
            'mg': 'MG', 'minas gerais': 'MG',
            'ms': 'MS', 'mato grosso do sul': 'MS',
            'mt': 'MT', 'mato grosso': 'MT',
            'pa': 'PA', 'pará': 'PA', 'para': 'PA',
            'pb': 'PB', 'paraíba': 'PB', 'paraiba': 'PB',
            'pe': 'PE', 'pernambuco': 'PE',
            'pi': 'PI', 'piauí': 'PI', 'piaui': 'PI',
            'pr': 'PR', 'paraná': 'PR', 'parana': 'PR',
            'rj': 'RJ', 'rio de janeiro': 'RJ',
            'rn': 'RN', 'rio grande do norte': 'RN',
            'ro': 'RO', 'rondônia': 'RO', 'rondonia': 'RO',
            'rr': 'RR', 'roraima': 'RR',
            'rs': 'RS', 'rio grande do sul': 'RS',
            'sc': 'SC', 'santa catarina': 'SC',
            'se': 'SE', 'sergipe': 'SE',
            'sp': 'SP', 'são paulo': 'SP', 'sao paulo': 'SP',
            'to': 'TO', 'tocantins': 'TO'
        }
        
        # 街道类型标准化
        self.street_type_mapping = {
            'r': 'Rua', 'rua': 'Rua',
            'av': 'Avenida', 'av.': 'Avenida', 'avenida': 'Avenida',
            'al': 'Alameda', 'al.': 'Alameda', 'alameda': 'Alameda',
            'trav': 'Travessa', 'trav.': 'Travessa', 'travessa': 'Travessa',
            'rod': 'Rodovia', 'rod.': 'Rodovia', 'rodovia': 'Rodovia',
            'est': 'Estrada', 'est.': 'Estrada', 'estrada': 'Estrada',
            'praça': 'Praça', 'praca': 'Praça', 'pc': 'Praça',
            'vl': 'Vila', 'vila': 'Vila',
            'jd': 'Jardim', 'jardim': 'Jardim',
            'blv': 'Boulevard', 'boulevard': 'Boulevard',
            'qs': 'Quadra', 'quadra': 'Quadra',
            'st': 'Setor', 'setor': 'Setor'
        }
        
        # 城市名称映射（可根据需要扩展）
        self.city_mapping = {
            'são paulo': 'São Paulo', 'sao paulo': 'São Paulo', 'sp': 'São Paulo',
            'rio de janeiro': 'Rio de Janeiro', 'rj': 'Rio de Janeiro',
            'belo horizonte': 'Belo Horizonte', 'bh': 'Belo Horizonte',
            'salvador': 'Salvador', 'ssa': 'Salvador',
            'brasília': 'Brasília', 'brasilia': 'Brasília', 'bsb': 'Brasília'
        }
        
        # 邮政编码模式 - 巴西格式: XXXXX-XXX
        self.zip_pattern = re.compile(r'\b(\d{5})-?(\d{3})\b')

    def clean_address(self, address: str) -> str:
        """清理地址，去除多余空格和特殊字符"""
        if not address or pd.isna(address):
            return ""
            
        # 转换为小写并去除首尾空格
        cleaned = str(address).lower().strip()
        
        # 替换多个连续空格为单个空格
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # 处理常见的分隔符问题
        cleaned = re.sub(r'[,\-\.;]+', ', ', cleaned)
        
        # 清理多余的空格和逗号组合
        cleaned = re.sub(r',\s*,', ',', cleaned)
        cleaned = re.sub(r'\s*,\s*', ', ', cleaned)
        
        # 去除特殊字符，只保留字母、数字、空格和基本标点
        cleaned = re.sub(r'[^\w\s,\.\-]', '', cleaned)
        
        # 再次清理空格
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned

    def standardize_street_type(self, street_part: str) -> str:
        """标准化街道类型"""
        words = street_part.split()
        if words and words[0].lower() in self.street_type_mapping:
            words[0] = self.street_type_mapping[words[0].lower()]
        return ' '.join(words)

    def extract_zip_code(self, address: str) -> Optional[str]:
        """从地址中提取邮政编码"""
        match = self.zip_pattern.search(address)
        if match:
            return f"{match.group(1)}-{match.group(2)}"
        return None

    def validate_and_format_zip_code(self, zip_code: str) -> Optional[str]:
        """验证和格式化邮政编码"""
        if not zip_code:
            return None
            
        # 移除非数字字符
        clean_zip = re.sub(r'\D', '', str(zip_code))
        
        if len(clean_zip) == 8:
            return f"{clean_zip[:5]}-{clean_zip[5:]}"
        elif len(clean_zip) == 5:
            return clean_zip
        else:
            return None

    def standardize_state(self, state: str) -> str:
        """标准化州名"""
        if not state:
            return ""
        clean_state = state.lower().strip()
        return self.state_mapping.get(clean_state, state.upper())

    def standardize_city(self, city: str) -> str:
        """标准化城市名"""
        if not city:
            return ""
        clean_city = city.lower().strip()
        return self.city_mapping.get(clean_city, city.title())

    def parse_address_components(self, address: str) -> Dict[str, str]:
        """
        解析地址组件
        巴西地址常见格式: [Rua/Av.] [Nome], [Número], [Bairro], [Cidade] - [Estado], [CEP]
        """
        components = {
            'logradouro': '',  # 街道地址
            'numero': '',      # 号码
            'bairro': '',      # 街区
            'cidade': '',      # 城市
            'estado': '',      # 州
            'cep': ''          # 邮政编码
        }
        
        # 提取邮政编码
        zip_code = self.extract_zip_code(address)
        if zip_code:
            components['cep'] = zip_code
            # 从地址中移除已识别的邮政编码
            address = self.zip_pattern.sub('', address)
        
        # 按逗号分割地址组件
        parts = [part.strip() for part in address.split(',') if part.strip()]
        
        if len(parts) >= 1:
            # 第一个部分通常是街道地址
            street_part = parts[0]
            # 标准化街道类型
            components['logradouro'] = self.standardize_street_type(street_part)
        
        if len(parts) >= 2:
            # 第二个部分可能是号码或街区
            second_part = parts[1]
            if re.search(r'\d', second_part):  # 包含数字，可能是号码
                components['numero'] = second_part
            else:
                components['bairro'] = second_part.title()
        
        if len(parts) >= 3:
            # 第三个部分可能是街区或城市
            third_part = parts[2]
            if not components['bairro'] and not re.search(r'\d', third_part):
                components['bairro'] = third_part.title()
        
        # 尝试从剩余部分提取城市和州
        remaining_parts = parts[3:] if len(parts) > 3 else []
        for part in remaining_parts:
            # 检查是否是州缩写（2个字母）
            if len(part.strip()) == 2 and part.strip().isalpha():
                components['estado'] = self.standardize_state(part.strip())
            elif len(part.split()) <= 3:  # 可能是城市名
                if not components['cidade']:
                    components['cidade'] = self.standardize_city(part)
        
        return components

    def complete_missing_data(self, components: Dict[str, str]) -> Dict[str, str]:
        """补全缺失的数据（基于已知信息）"""
        completed = components.copy()
        
        # 如果有邮政编码但缺少城市/州，可以在这里添加逻辑从数据库查询
        # 这里只是示例，实际应用中需要连接数据库或使用API
        if completed['cep'] and not completed['cidade']:
            # 示例：根据CEP前缀推断城市（实际需要完整数据库）
            cep_prefix = completed['cep'][:5]
            # 这里可以添加CEP到城市的映射逻辑
            
        # 如果有城市但缺少州，尝试推断
        if completed['cidade'] and not completed['estado']:
            city_lower = completed['cidade'].lower()
            if 'são paulo' in city_lower or 'sao paulo' in city_lower:
                completed['estado'] = 'SP'
            elif 'rio de janeiro' in city_lower:
                completed['estado'] = 'RJ'
            elif 'belo horizonte' in city_lower:
                completed['estado'] = 'MG'
            elif 'salvador' in city_lower:
                completed['estado'] = 'BA'
            elif 'brasília' in city_lower or 'brasilia' in city_lower:
                completed['estado'] = 'DF'
        
        return completed

    def format_brazilian_address(self, components: Dict[str, str]) -> str:
        """将地址组件格式化为标准巴西地址格式"""
        address_parts = []
        
        # 街道地址和号码
        if components['logradouro']:
            street_part = components['logradouro']
            if components['numero']:
                street_part += f", {components['numero']}"
            address_parts.append(street_part)
        
        # 街区
        if components['bairro']:
            address_parts.append(components['bairro'])
        
        # 城市和州
        city_state = []
        if components['cidade']:
            city_state.append(components['cidade'])
        if components['estado']:
            city_state.append(components['estado'])
        if city_state:
            address_parts.append(" - ".join(city_state))
        
        # 邮政编码
        if components['cep']:
            address_parts.append(components['cep'])
        
        return ", ".join(address_parts)

    def process_address(self, address: str) -> str:
        """处理单个地址的主函数"""
        try:
            if not address or pd.isna(address):
                return ""
            
            # 第一步：清理地址
            cleaned_address = self.clean_address(address)
            
            # 第二步：解析地址组件
            components = self.parse_address_components(cleaned_address)
            
            # 第三步：补全缺失数据
            completed_components = self.complete_missing_data(components)
            
            # 第四步：格式化标准地址
            formatted_address = self.format_brazilian_address(completed_components)
            
            return formatted_address
            
        except Exception as e:
            print(f"处理地址时出错 '{address}': {e}")
            return str(address)  # 返回原始地址作为备选

def process_excel_addresses(input_file: str, output_file: str, address_column: str = 'endereço'):
    """
    处理Excel文件中的葡萄牙语地址
    
    参数:
        input_file: 输入Excel文件路径
        output_file: 输出Excel文件路径  
        address_column: 包含地址的列名
    """
    try:
        # 读取Excel文件
        df = pd.read_excel(input_file)
        
        # 检查地址列是否存在
        if address_column not in df.columns:
            raise ValueError(f"列 '{address_column}' 在Excel文件中不存在")
        
        # 初始化地址格式化器
        formatter = PortugueseAddressFormatter()
        
        # 处理所有地址
        print("正在处理地址...")
        df['endereço_formatado'] = df[address_column].apply(formatter.process_address)
        
        # 保存结果
        df.to_excel(output_file, index=False)
        print(f"处理完成! 结果已保存到: {output_file}")
        
        # 显示统计信息
        original_count = len(df)
        processed_count = len(df[df['endereço_formatado'].notna() & (df['endereço_formatado'] != '')])
        print(f"处理统计: {processed_count}/{original_count} 个地址成功处理")
        
        return df
        
    except Exception as e:
        print(f"处理Excel文件时出错: {e}")
        return None

# 使用示例
def main():
    """使用示例"""
    # 示例地址列表
    sample_addresses = [
        "r das flores, 123, centro, sao paulo, sp, 01234-567",
        "av paulista, 1000, bela vista, sao paulo, 01310-100",
        "rua xv de novembro, 200, centro, rio de janeiro, rj",
        "travessa da paz, 45, liberdade, salvador, ba, 40050-000",
        "alameda santos, 500, jardins, 01418-200",
        "rodovia dos imigrantes, km 25, praia grande, sp"
    ]
    
    formatter = PortugueseAddressFormatter()
    
    print("地址格式化示例:")
    print("=" * 50)
    
    for i, address in enumerate(sample_addresses, 1):
        print(f"\n{i}. 原始地址: {address}")
        formatted = formatter.process_address(address)
        print(f"   格式化后: {formatted}")
    
    print("\n" + "=" * 50)
    
    # Excel处理示例（需要实际文件）
    # process_excel_addresses('input.xlsx', 'output.xlsx', 'endereço')

if __name__ == "__main__":
    main()