import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import pyautogui
import pygetwindow as gw
import pytesseract
import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from difflib import get_close_matches
from modelos import DQN
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageGrab

#Load do modelo
def load_or_create_state_dict(model, nome_modelo):
    checkpoint_path = f'modelo\{nome_modelo}'
    print(checkpoint_path)
    try:
        # Load the model state dictionary from the file
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        print("Modelo carregado com sucesso.")
                
    # If the file is not found, create a new state dictionary for the model
    except FileNotFoundError:
        print("Model file not found. Creating a new model.")
        state_dict = model.state_dict()
        torch.save(state_dict, checkpoint_path)
        print(f"New model state saved to {checkpoint_path}")

    model.load_state_dict(state_dict)

    return model

#funções de Modelo
def calcular_target(Q, Q_proximo, recompensa, gamma,learningrate):
    return Q + learningrate * (recompensa + gamma * torch.max(Q_proximo) - Q)

def treina_dqn(modelo, entradas, recompensas, Q_proximo, taxa_aprendizado):
    gamma = 0.9
    
    lossfn = nn.CrossEntropyLoss()
        
    otimizador = optim.Adam(params=modelo.parameters(), lr=taxa_aprendizado)
    
    #next state
    Q_proximo = torch.tensor(Q_proximo, dtype=torch.float32)
    Q_proximo = Q_proximo.clone().detach().requires_grad_(True)
    Q_proximo = Q_proximo.unsqueeze(0)
    Q_proximo = modelo(Q_proximo)
    
    #atual state
    Q = entradas.clone().detach().requires_grad_(True)
    Q_atual = Q

    #Reward
    recompensas = recompensas['reward']
    recompensas_tensores = torch.tensor(np.array(recompensas), dtype=torch.float32)
    
    modelo.train()
    target_Q = calcular_target(Q,Q_proximo,recompensas_tensores,gamma,taxa_aprendizado)
    loss = lossfn(Q_atual, target_Q)
    
    # Atualize os parâmetros do modelo
    otimizador.zero_grad()
    loss.backward(retain_graph=True)
    otimizador.step()

    # Atualize o estado atual
    percas = loss.item()
    return modelo, percas

#função reset env
def pressionar_f2_a_cada_60_segundos(segundos,rest,loop):
    if segundos % rest ==0:
        # Pressiona o botão F2
        pyautogui.press('F2')
        segundos += 1
        loop +=1
        print('loop:', loop)
    else:
        segundos += 1
        
    return segundos,loop

def salvascore(score,segundos,rest,recompensanoloop):
    if (segundos +1) % rest ==0:
        recompensanoloop.append(score)
        score = 0
    else:
        score
    return score
    
#aplica o modelo e aperta o botão

def calculate_epsilon(decay_rate,contador_frames):
    # Definir parâmetros
    initial_epsilon = 1.0  # Valor inicial de epsilon
    final_epsilon = 0.01    # Valor final de epsilon
    
    if contador_frames == 0:
        return initial_epsilon
    else:
        epsilon = final_epsilon + (initial_epsilon - final_epsilon) * np.exp(-decay_rate * contador_frames)
    
    if contador_frames % 1000 == 0:
        print('epsilon',epsilon)
    
    return epsilon

def escolher_tecla_e_botao(modelo,frame_atual, mapeamento,contador_frames,decay_rate):
    epsilon = calculate_epsilon(decay_rate,contador_frames)
    # Ação aleatória para exploração
    if np.random.rand() < epsilon:
        Greed = 1
        Nxt_indice = np.random.choice(list(mapeamento.keys()))
        tecla_seta = mapeamento[Nxt_indice][0]
        botao = mapeamento[Nxt_indice][1]
        saida_modelo_logits = torch.zeros(11, dtype=torch.float)
        saida_modelo_logits = torch.unsqueeze(saida_modelo_logits, dim=0)
        saida_modelo_logits[0, Nxt_indice] = 1

    # Utilização do modelo
    else:
        Greed = 0

        Xmodelo = torch.tensor(frame_atual, dtype=torch.float32)
        Xmodelo = Xmodelo.unsqueeze(0)
            
        modelo.eval()
        with torch.no_grad():
            saida_modelo_logits = modelo(Xmodelo)
            f = nn.Softmax(dim=1)
            saida_modelo_logits = f(saida_modelo_logits)

        # Convertendo as saídas do modelo para índices inteiros
        indices = torch.Tensor.numpy(saida_modelo_logits)
        Nxt_indice = np.argmax(indices, axis=1)
        Nxt_indice = Nxt_indice[0]

        # Obtendo as teclas e botões correspondentes aos índices
        tecla_seta = mapeamento[Nxt_indice][0]
        #print(tecla_seta)
        botao = mapeamento[Nxt_indice][1]
        
    return tecla_seta, botao, saida_modelo_logits, Nxt_indice,Greed

def executar_acao(tecla_seta, botao):
    def pressionar_liberar(tecla_seta, botao):
        pyautogui.keyDown(tecla_seta)
        pyautogui.keyDown(botao)
        #time.sleep(0.001)
        pyautogui.keyUp(tecla_seta)
        pyautogui.keyUp(botao)
        
    def pressionar_liberar2(tecla):
        pyautogui.keyDown(tecla)
        time.sleep(0.001)
        pyautogui.keyUp(tecla)

    # Executar ação para tecla de seta
    if tecla_seta is not None and botao is not None:
        pressionar_liberar(tecla_seta, botao)
    elif tecla_seta is not None: 
        pressionar_liberar2(tecla_seta)
    elif botao is not None:
        pressionar_liberar2(botao)
    else:
        None

#Extrai os dados da imagem
def processar_imagem(screenshot):
    # Abrir a imagem
    imagem = screenshot
    
    # Obter as dimensões da imagem
    largura, altura = imagem.size
    
    # Calcular a altura da porção que você deseja manter (80% do original)
    nova_altura = int(0.9 * altura)
    nova_altura2 = int(0.2 * altura)
    corte_cima = altura - nova_altura

    # Calcular a largura da porção a ser mantida (70% do original)
    nova_largura = int(0.18 * largura)
    nova_largura2 = int(0.5 * largura)

    # Definir as coordenadas da área a ser mantida
    coordenadas_corte = (nova_largura, corte_cima, nova_largura2, nova_altura2)
 
    
    # Cortar a imagem
    imagem = imagem.crop(coordenadas_corte)

    
    width, height = imagem.size
    
    # Duplica o tamanho da imagem
    new_width = width * 2
    new_height = height * 2

    # Redimensiona a imagem
    imagem = imagem.resize((new_width, new_height))
    
    # Converter para escala de cinza
    imagem = imagem.convert("L")
    
    # Inverter as cores
    img = ImageOps.invert(imagem)
    
    #melhora contraste e brilho
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.5)
    
    # Extrair texto da imagem
    texto_extraido = pytesseract.image_to_string(img, config='--psm 6 --oem 3', lang='eng')
    
    
    return texto_extraido

def extrair_dados(texto_extraidoE, valores_anteriores, Nxt_indice, ultimos_indices,repet,vida):
   # Create a defaultdict with default value 0
    dados_dict = defaultdict(int, valores_anteriores)
    
    palavras_chave = ['bonus', 'score', 'reward']
    #Verifica o fator punitivo por muita repetição
    remov = calcular_valor(Nxt_indice, ultimos_indices, repet)
    
    for linha in texto_extraidoE.split('\n'):
        match = re.match(r'\b(\w+)\s+(\d+)\b', linha)
        if match:
            palavra_chave = get_close_matches(match.group(1).lower(), palavras_chave, n=1, cutoff=0.3)
            if palavra_chave:
                chave = palavra_chave[0]
                valor = int(match.group(2))
                dados_dict[chave] = valor
            else:
                chave = match.group(1).lower()
                dados_dict[chave] = valores_anteriores.get(chave, 0)

    if 'bonus' in dados_dict or 'score' in dados_dict:
        reward = min(dados_dict.get('score', 0), 100000)
        dados_dict['reward'] = reward
    else:
        dados_dict['reward'] = 0

    reward = {}
    for key in valores_anteriores.keys():
        if key == 'reward' and key in dados_dict:
            reward[key] = max(min((max(dados_dict[key] - valores_anteriores[key], 0) / 30) - remov - vida, 10), -5)

    bonus_T = False  # Inicializa como False
    if not valores_anteriores:
        bonus_T = False
    elif "bonus" in dados_dict:
        bonus_T = (dados_dict['bonus'] != valores_anteriores.get('bonus', None))
    else: False

    return reward, dados_dict,ultimos_indices, bonus_T

def maxscore(score, valores_anteriores,segundos,recompensanoloop,rest,vida,loop):
    score_anterior = valores_anteriores.get('score', 0)
    segundos = segundos
    if score_anterior > score:
        score = score_anterior
    else:
        score = score
        
    if vida <= 0.1:
        pyautogui.press('F2')
        segundos = 1
        #print('vida')
        #print(score)
        recompensanoloop.append(score)
        loop +=1
        if loop % 100 == 0:
            print('loop:', loop)
        score = 0
        
        
    score = salvascore(score,segundos,rest,recompensanoloop)
    return score, loop,segundos
    
#Funções que verificam modificadores de recompensa
def Vida(screenshot,vida):
    #pega região que tem a vida no jogo
    imagem = screenshot.crop((int(0.16 * screenshot.width), screenshot.height - int(0.78 * screenshot.height), int(0.48 * screenshot.width), int(0.24 * screenshot.height)))

    #Verifica % da vida
    porcentagem_preenchimento = preenchimento(imagem)

    if porcentagem_preenchimento < vida:
        difvida = 1-(vida - porcentagem_preenchimento)
    else:
        difvida = 0
    
    vida = porcentagem_preenchimento
    
    return vida,difvida

def preenchimento(imagem): 
    # Carregar a imagem e converter para escala de cinza
    imagem_pb = imagem.convert("L")
    imagem_array = np.array(imagem_pb)
    # Calcular o índice de início e fim para a região central
    largura, altura  = imagem_array.shape
    coluna_central = imagem_array[largura // 2,6:-6]
    pixels_preenchidos = np.count_nonzero(coluna_central > 120)
    porcentagem_preenchimento = (pixels_preenchidos / len(coluna_central))

    return porcentagem_preenchimento

def calcular_valor(indice_atual, ultimos_indices,repet):
    # Adiciona o índice atual à lista de últimos índices
    ultimos_indices.append(indice_atual)
    if len(ultimos_indices) >5:
        ultimos_indices.pop(0)
    
    # Verifica se os últimos 3 índices são iguais ao índice atual
    if ultimos_indices[-3:] == [4, 4, 4]:
        return 0
    elif len(ultimos_indices) >= 3:
        if ultimos_indices[-3] == ultimos_indices[-2] == ultimos_indices[-1]:
            return repet
        else:
            return 0
    else:
        return 0

#transforma imagem em array e junta em frames
def getscreen():
    janela = gw.getWindowsWithTitle('snes9x')[0]
    x, y, width, height = janela.left, janela.top, janela.width, janela.height
    screenshot = ImageGrab.grab(bbox=(x, y, x + width, y + height))
    
    return screenshot

def criarray(screenshot,x_pixel, y_pixel):
# Redimensionar a imagem

    width, height = screenshot.size
 
    # Setting the points for cropped image
    left = width / 75
    top = height / 3.7
    right = width/ 1.02
    bottom = height/1.03

    imagem = screenshot.crop((left, top, right, bottom))

    imagem_redimensionada = imagem.resize((x_pixel, y_pixel))
    

    # Converter a imagem para escala de cinza (preto e branco)
    imagem_preto_branco = imagem_redimensionada.convert("L")

    # Converter a imagem para um array numpy
    array_imagem = np.array(imagem_preto_branco)
    
    return array_imagem

def juntaframes(x_pixel, y_pixel,num_screenshots):
    frames_sequence = []
    
    for _ in range(num_screenshots):
        screenshot = getscreen()
        array_imagem = criarray(screenshot, x_pixel, y_pixel)
        frames_sequence.append(array_imagem)
        
    frame = np.reshape(frames_sequence, (num_screenshots, x_pixel, y_pixel))

    return frame,screenshot

#Verifica se é a primeira iamgem        
def check_or_capture_screenshot(x_pixel, y_pixel, nxt_frame,num_screenshots):
    if nxt_frame is not None:
        # Se o frame de imagem já existe, use-o
        frame_atual = nxt_frame
    else:
        # Se o frame de imagem não existe, capture um screenshot
        frame_atual,screenshot = juntaframes(x_pixel, y_pixel,num_screenshots)

    return frame_atual

#plotar
def plotar(perdas,Meanrecompensa,recompensanoloop):
    fig, ax1 = plt.subplots()

    # Plotar as perdas no primeiro subplot
    ax1.plot(perdas, label='Perdas', color='blue')
    ax1.set_xlabel('Tempo')
    ax1.set_ylabel('Perda', color='blue')
    ax1.tick_params('y', colors='blue')
    ax1.legend(loc='upper left')

    # Plotar a média das recompensas no segundo subplot
    fig2, ax2 = plt.subplots()
    
    ax2.plot(Meanrecompensa, label='Média das Recompensas', color='orange')
    ax2.set_xlabel('Tempo')
    ax2.set_ylabel('Média das Recompensas', color='orange')
    ax2.tick_params('y', colors='orange')
    ax2.legend(loc='upper right')

    fig3, ax3 = plt.subplots()
    
    ax3.plot(recompensanoloop, label='recompensanoloop', color='red')
    ax3.set_xlabel('loop')
    ax3.set_ylabel('recompensanoloop', color='red')
    ax3.tick_params('y', colors='red')
    ax3.legend(loc='upper right')
    ax3.set_ylim(bottom=0)

    # Mostrar os subplots
    plt.show()

def quebratempo(start_time, local):
    end_time = time.time()
    total_elapsed_time = end_time - start_time
    print(local,total_elapsed_time)
    start_time = time.time()

    return start_time

def salvametrica(perdas,mean_recompensa,indice,greed,nome_arquivo):

    # Verifica se o arquivo já existe
    try:
        # Tenta carregar os dados existentes
        df_existente = pd.read_csv(f'csv\{nome_arquivo}')
        
        # Se o arquivo não estiver vazio, concatena os novos dados
        if not df_existente.empty:
            df_novo = pd.DataFrame({'Perdas': perdas, 'Mean Recompensa': mean_recompensa,'indice': indice,'Greed': greed})
            df_concatenado = pd.concat([df_existente, df_novo], ignore_index=True)
        else:
            # Se o arquivo estiver vazio, usa o DataFrame novo
            df_concatenado = pd.DataFrame({'Perdas': perdas, 'Mean Recompensa': mean_recompensa,'indice': indice,'Greed': greed})
    except FileNotFoundError:
        # Se o arquivo não existir, cria um novo DataFrame
        df_concatenado = pd.DataFrame({'Perdas': perdas, 'Mean Recompensa': mean_recompensa,'indice': indice,'Greed': greed})

    # Salva o DataFrame no arquivo
    df_concatenado.to_csv(f'csv\{nome_arquivo}', index=False)
    
def statusmodelo(modelo):
    printa = False
    if printa ==True:
        for name, param in modelo.named_parameters():
            if 'weight' in name:
                weights = param.data.flatten()
                print(f'Pesos de {name}:')
                print(f'Média: {weights.mean()}, Mínimo: {weights.min()}, Máximo: {weights.max()}, Desvio Padrão: {weights.std()}')
            elif 'bias' in name:
                biases = param.data.flatten()
                print(f'Biases de {name}:')
                print(f'Média: {biases.mean()}, Mínimo: {biases.min()}, Máximo: {biases.max()}, Desvio Padrão: {biases.std()}')    
        
#função junta

def primeira_parte(x_pixel, y_pixel, nxt_frame, num_screenshots,modelo, mapeamento, contador_frames,decay_rate):
    frame_atual = check_or_capture_screenshot(x_pixel, y_pixel, nxt_frame, num_screenshots)
    tecla_seta, botao, saida_modelo_logits, Nxt_indice,Greed = escolher_tecla_e_botao(modelo, frame_atual, mapeamento,contador_frames,decay_rate)
    executar_acao(tecla_seta, botao)
    
    return saida_modelo_logits, Nxt_indice,Greed

def segunda_parte(x_pixel, y_pixel,num_screenshots,vida, valores_anteriores, Nxt_indice, ultimos_indices, repet,contador_frames, score, loop, segundos,recompensanoloop, rest):
    
    nxt_frame,screenshot = juntaframes(x_pixel, y_pixel, num_screenshots)
    
    texto_extraidoE  = processar_imagem(screenshot)
    
    vida, difvida = Vida(screenshot, vida)
    
    reward, valores_anteriores, ultimos_indices, bonus_T = extrair_dados(texto_extraidoE, valores_anteriores, Nxt_indice, ultimos_indices, repet, difvida)  
    
    contador_frames += 1
    
    score, loop, segundos = maxscore(score, valores_anteriores, segundos, recompensanoloop, rest, vida, loop)
    
    return contador_frames, score, loop, segundos,nxt_frame,reward,bonus_T,valores_anteriores,ultimos_indices,vida

def agente(segundos, rest, loop, x_pixel, y_pixel, nxt_frame, modelo, mapeamento, valores_anteriores, ultimos_indices, repet, contador_frames, score, recompensanoloop, vida,num_screenshots,decay_rate):
    # Criar uma fila para comunicação entre as threads
    segundos, loop = pressionar_f2_a_cada_60_segundos(segundos, rest, loop)
    
    saida_modelo_logits, Nxt_indice,Greed = primeira_parte(x_pixel, y_pixel, nxt_frame, num_screenshots,modelo, mapeamento, contador_frames,decay_rate)
    
    contador_frames, score, loop, segundos,nxt_frame,reward,bonus_T,valores_anteriores,ultimos_indices,vida = segunda_parte(x_pixel, y_pixel,num_screenshots,vida, valores_anteriores, Nxt_indice, ultimos_indices, repet,contador_frames, score, loop, segundos,recompensanoloop, rest)
    
    return segundos, loop, nxt_frame, contador_frames, score, valores_anteriores, ultimos_indices,saida_modelo_logits,reward,bonus_T,vida,Greed,Nxt_indice

def treino(bonus_T,modelo, entradas, recompensas, Q_proximo, perdas,Meanrecompensa,Greed,Nxt_indice,greed,indice,taxa_aprendizado):
    if bonus_T == True:
        modelo, percas_atual = treina_dqn(modelo, entradas, recompensas, Q_proximo,taxa_aprendizado)
        
        perdas.append(percas_atual)
        Meanrecompensa.append(recompensas['reward'])
        greed.append(Greed)
        indice.append(Nxt_indice)

    else:
        pyautogui.keyDown('x')
        time.sleep(0.005)
        pyautogui.keyUp('x')
        
    return modelo

def maintreino(contador_frames,tx,decay_rate,nome_arquivo,nome_modelo,maxloops,num_screenshots,xy_pixel,rest):
    # Inicializar o DataFrame fora do loop
    
    mapeamento = {
                    7: ('up', 'a'),
                    1: ('up', 'z'),
                    9: (None, 'a'),
                    0: (None, 'z'),
                    3: (None, 'x'),
                    5: ('up', 'x'),
                    10: ('down', None),
                    8: ('left', None),
                    2: ('right', None),
                    6: ('up', None),
                    4: (None, None)
    }
    
    #hiperparametros da rede
    kernel = 3
    loop = 0
        
    #valores para recompensa
    repet = 0.5
    
    #saida
    mapsize = len(mapeamento)
    
    #avaliações
    perdas = []
    Meanrecompensa = []
    tempomedio = []
    recompensanoloop = []
    greed = []
    indice = []

    #variaveis
    segundos = 0
    score = 0
    vida= 0
    
    #Reward
    valores_anteriores = {}
    ultimos_indices = []  
     
    #entrada
    
    Entrada = num_screenshots
    x_pixel = xy_pixel
    y_pixel = xy_pixel
    
    Nxt_indice = 0
    nxt_frame = None
    
    modelo = DQN(Entrada,mapsize,kernel)
    
    statusmodelo(modelo)
    
    modelo = load_or_create_state_dict(modelo,nome_modelo)
    
    try:
        while True:
            start_time = time.time()
            
            segundos, loop, Q_proximo, contador_frames, score, valores_anteriores, ultimos_indices,entradas,recompensas,bonus_T,vida,Greed,Nxt_indice  = agente(segundos, rest, loop, x_pixel, y_pixel, nxt_frame, modelo, mapeamento, valores_anteriores, ultimos_indices, repet, contador_frames, score, recompensanoloop, vida,num_screenshots,decay_rate)
            
            modelo = treino(bonus_T,modelo, entradas, recompensas, Q_proximo, perdas,Meanrecompensa,Greed,Nxt_indice,greed,indice, taxa_aprendizado=tx) 
            
            end_time = time.time()
            total_elapsed_time = end_time - start_time
            
            tempomedio.append(total_elapsed_time)

            if loop >= maxloops:
                raise KeyboardInterrupt

    except (KeyboardInterrupt, Exception) as e:
        if isinstance(e, KeyboardInterrupt):
            print("\nPrograma interrompido pelo usuário.")
        else:
            print(f"Ocorreu uma exceção não tratada: {e}")

        print(f"a contagem acabou no {contador_frames}...")
        print("Tempo medio:", sum(tempomedio)/len(tempomedio) )
        print("Meanrecompensa medio:", sum(Meanrecompensa)/len(Meanrecompensa) )
        
        statusmodelo(modelo)
        
        state = modelo.state_dict()
        
        torch.save(state, f'modelo\{nome_modelo}')
        print('modelo salvo')
        
        salvametrica(perdas,Meanrecompensa,indice,greed,nome_arquivo)
                
        plotar(perdas,Meanrecompensa,recompensanoloop)
                 