import sys
# Importa todas as funções e variáveis necessárias do nosso módulo
from modelo_ag import (
    rodar_ag, 
    calcular_score_inicial, 
    PONTUACAO_RENDA, 
    FATOR_EMPREGO, 
    PONTUACAO_HISTORICO, 
    PONTUACAO_TEMPO,
    LIMITE_MAX_NOVO_CLIENTE
)

# Mapeamentos para o usuário
MAPA_RENDA = {'1': 'Baixa (até 1 SM)', '2': 'Média (1-3 SM)', '3': 'Alta (3+ SM)'}
MAPA_EMPREGO = {'1': 'Informal', '2': 'Formal'}
MAPA_HISTORICO = {'1': 'Novo Cliente', '2': 'Bom Pagador', '3': 'Mau Pagador (quitado)'}
MAPA_TEMPO = {'1': 'Novo (até 1 ano)', '2': '1-5 anos', '3': '5+ anos'}


def get_input(prompt, options_map, options_dict):
    """ Função auxiliar para coletar e validar input numérico. """
    print(prompt)
    for k, v in options_map.items():
        print(f"  [{k}] - {v}")
    key = input("Opção: ").strip()
    if key not in options_dict:
        print("❌ Erro: Opção inválida.")
        return None
    return key

def coletar_input():
    """ Coleta as 5 informações do cliente via input. """
    print("\n--- Cadastro do Cliente ---")
    
    # 1. Histórico de Pagamento (Define o Fluxo)
    historico_key = get_input("Histórico de Pagamento Interno:", MAPA_HISTORICO, PONTUACAO_HISTORICO)
    if historico_key is None: return None

    # 4. Tempo de Loja (Condicional)
    if historico_key == '1': # 'Novo Cliente'
        tempo_loja_key = '1'
        print(f"Tempo de Loja: [1] - Novo (Assumido pelo Histórico)")
    else:
        tempo_loja_key = get_input("Tempo de Cliente na Loja:", MAPA_TEMPO, PONTUACAO_TEMPO)
        if tempo_loja_key is None: return None
    
    # 2. Renda Estimada
    renda_cat_key = get_input("Renda Estimada (Capacidade de Pagamento):", MAPA_RENDA, PONTUACAO_RENDA)
    if renda_cat_key is None: return None

    # 3. Tipo de Emprego
    tipo_emprego_key = get_input("Tipo de Emprego (Estabilidade da Renda):", MAPA_EMPREGO, FATOR_EMPREGO)
    if tipo_emprego_key is None: return None
        
    # 5. Valor da Compra
    try:
        print("Valor da Compra (R$): ")
        valor_compra = float(input().replace(',', '.'))
        if valor_compra <= 0: raise ValueError
    except ValueError:
        print("❌ Erro: Valor da compra deve ser um número positivo.")
        return None

    return renda_cat_key, tipo_emprego_key, historico_key, tempo_loja_key, valor_compra

def tomar_decisao(melhor_c):
    """ Função principal que integra o Score (IA/ML) e a Decisão (AG), aplicando a regra do teto. """
    print("\n=======================================================")
    print("  SISTEMA HÍBRIDO DE CRÉDITO PARA MPE (IA + AG)")
    print("=======================================================")
    print(f"1. A POLÍTICA OTIMIZADA DO AG: C = {melhor_c:.4f}.")

    cliente_data = coletar_input()
    if cliente_data is None: return

    renda_cat_key, tipo_emprego_key, historico_key, tempo_loja_key, valor_compra = cliente_data

    # 2. CALCULAR O SCORE (IA/ML)
    score_cliente = calcular_score_inicial(renda_cat_key, tipo_emprego_key, historico_key, tempo_loja_key)
    
    # 3. CÁLCULO DO LIMITE E APLICAÇÃO DA REGRA DE NEGÓCIO (TETO)
    limite_calculado_ag = score_cliente / melhor_c
    limite_aplicado = limite_calculado_ag
    
    # Aplica a Regra de Teto para Clientes Novos
    is_novo_cliente = (historico_key == '1')
    if is_novo_cliente:
        if limite_calculado_ag > LIMITE_MAX_NOVO_CLIENTE:
            limite_aplicado = LIMITE_MAX_NOVO_CLIENTE
        
    # DECISÃO FINAL: 
    decisao = "APROVAR" if valor_compra <= limite_aplicado else "NEGAR"

    racio = score_cliente / valor_compra

    print("\n--- RESULTADO DA ANÁLISE ---")
    print(f"SCORE INICIAL (PD): {score_cliente:.2f}")
    print(f"VALOR DA COMPRA SOLICITADA: R$ {valor_compra:,.2f}")
    print(f"RÁCIO (SCORE / VALOR): {racio:.4f}")
    
    print("\n------------------------------------------------------")
    if decisao == "APROVAR":
        print(f"✅ DECISÃO FINAL: APROVAR A VENDA DE R$ {valor_compra:,.2f}.")
        print(f"   (Limite Total Aplicado: R$ {limite_aplicado:,.2f})")
    else:
        print(f"❌ DECISÃO FINAL: NEGAR A VENDA DE R$ {valor_compra:,.2f}.")
        print(f"   (O Score só permite um limite máximo de R$ {limite_aplicado:,.2f})")
    
    if is_novo_cliente:
        print(f"\n   *NOTA: Para clientes novos, o limite máximo é fixado em R$ {LIMITE_MAX_NOVO_CLIENTE:,.2f} (Regra de Negócio).")
    print("------------------------------------------------------")

# ====================================================================
# EXECUÇÃO DO SISTEMA
# ====================================================================

if __name__ == "__main__":
    # 0. Roda o AG uma vez para definir a política 'C'
    print("--- ⚙️ CALIBRANDO A POLÍTICA DE CRÉDITO DA LOJA VIA ALGORITMO GENÉTICO... ⚙️ ---")
    melhor_coeficiente_c_otimizado = rodar_ag()
    print(f"POLÍTICA OTIMIZADA ENCONTRADA: C = {melhor_coeficiente_c_otimizado:.4f}")

    # 1. Roda a interface para testar a decisão
    tomar_decisao(melhor_coeficiente_c_otimizado)