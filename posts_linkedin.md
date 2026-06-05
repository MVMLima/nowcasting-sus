# Posts LinkedIn — Nowcasting-SUS e Consultoria

---

## Post 1 — Lançamento do pacote nowcasting-sus

**Título:** Agora tem pacote Python pra corrigir atraso do SINAN

Eu passei os últimos meses transformando uma coisa que só rodava no meu notebook em algo que qualquer equipe de vigilância pode instalar com um `pip install`.

O nowcasting-sus está no PyPI. Open source, MIT, pronto pra uso.

**O problema que ele resolve é simples:** os dados do SINAN chegam atrasados. Não por falha do sistema — é da natureza da notificação. O caso acontece hoje, mas leva dias ou semanas até aparecer na base. Se você toma decisão olhando o número bruto, vai sempre subestimar o que está acontecendo agora.

O nowcasting-sus usa PyMC pra modelar o padrão de atraso histórico e estimar os casos prováveis em tempo real. Não é adivinhação. É o mesmo método que usei na Bahia em 2024 e que acertou dentro do IC em todas as semanas durante 6 meses.

**O que vem no pacote:**
- Modelo base RW1 + NegativeBinomial
- Modelo com efeito de dia da semana
- CLI que gera boletim HTML com um comando
- Interface limpa pra quem quer só plugar os dados e rodar

`pip install nowcasting-sus` e o repositório está em github.com/MVMLima/nowcasting-sus.

O lance simbólico pra mim: esse código nasceu dentro da SESACRE, rodou em produção durante a epidemia de chikungunya na Bahia, ganhou o Prêmio Dionísio Herrera na EXPOEPI 2026, e agora qualquer estado ou município pode usar sem custo.

Se você trabalha com vigilância e já teve que explicar pro gestor por que o número de casos "pulou" quando fechou a semana — esse pacote é pra você.

**Como sua equipe lida hoje com o atraso de notificação do SINAN?**

#Nowcasting #SUS #Epidemiologia #PyMC #SaúdePública

---

## Post 2 — Oferta de treinamento + consultoria

**Título:** Quer implantar nowcasting na sua secretaria?

Depois que o nowcasting-sus virou pacote público, comecei a receber mensagens de equipes de todo o país perguntando a mesma coisa: "como a gente coloca isso pra rodar aqui?"

A resposta curta: depende. Cada estado tem um fluxo de dados diferente, uma equipe com nível técnico diferente, agravos prioritários diferentes. Não existe solução genérica que funcione sem adaptação.

Por isso estou oferecendo **consultoria e treinamento** pra implantação real, em campo.

**O que entrego:**
- Configuração completa do pipeline de nowcasting — da coleta dos dados ao boletim semanal automático
- Treinamento prático pra equipe (remoto ou presencial) — interpretação de curvas, validação de resultados, operação do sistema
- Dashboard ou boletim automatizado que roda sozinho
- Acompanhamento por 1 a 3 meses pra garantir que o modelo continua funcionando

Não é curso teórico com slides. É o mesmo processo que usei pra implantar na SESACRE e que rendeu o prêmio na EXPOEPI.

**Público:** secretarias estaduais e municipais de saúde, OPAS, Fiocruz, ONGs — qualquer equipe que queira sair do dado atrasado pra estimativa em tempo real.

Interessado? Me manda uma mensagem ou email (mvmlima@hotmail.com) que a gente conversa sobre o cenário de vocês.

**Qual o maior gargalo que sua equipe enfrenta pra usar nowcasting na rotina?**

#VigilânciaEmSaúde #Nowcasting #SUS #Capacitação #CiênciaDeDados
