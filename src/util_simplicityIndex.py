
# ================================================================================================================================================================================================
# INSTRUCTIONS 
""""
To use, simply import this script and use `def composite_index(text)`, where text is a string of the document. 
"""


# ================================================================================================================================================================================================
# FUNCTIONS 

import spacy
import pandas as pd 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math 

# Tokenize 
nlp = spacy.load("en_core_web_sm")

def tokenize_sentences(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def tokenize_words(sentence):
    doc = nlp(sentence.lower())
    return [token.text for token in doc if not token.is_punct and not token.is_space]

def syllable_count(word: str) -> int :
    """
    Returns the count of syllables in a word

    # Notes
    Uses basic heuristics: 
    1. If the first letter of the word is a vowel, increment the syllable count.
    2. For each vowel encountered, increment the syllable count if the preceding character is not a vowel.
    3. If the word ends with the letter 'e', decrement the syllable count.
    4. If the count is still zero after processing the entire word, increment it by 1, assuming there's at least one syllable in every word.
    """
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for i in range(1, len(word)):
        if word[i] in vowels and word[i - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count

def flesch_reading_ease(text):
    """
    Flesh_Reading_Ease = 206.835 - (1.015 x ASL) - (84.6 x ASW)
    # Output 
    range for Flesch Reading Ease is between 0 and 100ish, with higher scores indicating easier readability.
    """
    sentences = tokenize_sentences(text)
    sentences = [s for s in sentences if len(tokenize_words(s)) >= 2]
    words = tokenize_words(text)
    total_sentences = len(sentences)
    total_words = len(words)
    total_syllables = sum([syllable_count(word) for word in words])

    if total_words == 0 or total_sentences == 0:
        return 0

    avg_sentence_length = total_words / total_sentences
    avg_syllables = total_syllables / total_words

    return 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables

def dale_chall(text):
    """
    Raw score = 0.1579 x (PDW) + 0.0496 x ASL + 3.6365
    Where PDW = Percentage of difficult words not on the Dale-Chall word list
    ASL = Average sentence length (number of words divided by number of sentences)

    # Output
    4.9 or lower: easily understood by an average 4th-grade student or lower
    5.0–5.9: easily understood by an average 5th or 6th-grade student
    6.0–6.9: easily understood by an average 7th or 8th-grade student
    7.0–7.9: easily understood by an average 9th or 10th-grade student
    8.0–8.9: easily understood by an average 11th or 12th-grade student
    9.0–9.9: easily understood by an average college student
    10.0 and higher: easily understood by college graduates.
    """
    easy_words_file = "./data/dale_chall_easy_word_list.txt"  # Replace this with the file path containing the 3,000 easy words
    with open(easy_words_file, "r") as f:
        easy_words = set(f.read().lower().split())

    sentences = tokenize_sentences(text)
    sentences = [s for s in sentences if len(tokenize_words(s)) >= 2]
    words = tokenize_words(text)
    total_sentences = len(sentences)
    total_words = len(words)
    difficult_words = sum([1 for word in words if word not in easy_words])

    if total_words == 0 or total_sentences == 0:
        return 0

    avg_sentence_length = total_words / total_sentences
    percent_difficult_words = (difficult_words / total_words) * 100

    return 0.1579 * percent_difficult_words + 0.0496 * avg_sentence_length

def ari(text):
    """
    ARI = 4.71 * (characters / words) + 0.5 * (words / sentences) - 21.43

    #Output 
    -6.0 to 0.9: easily understood by an average 4th-grade student or lower
    1.0 - 1.9: easily understood by an average 5th or 6th-grade student
    2.0 - 2.9: easily understood by an average 7th or 8th-grade student
    3.0 - 3.9: easily understood by an average 9th or 10th-grade student
    4.0 - 4.9: easily understood by an average 11th or 12th-grade student
    5.0 - 5.9: easily understood by an average college student 
    6.0 - 6.9: easily understood by graduate students
    7.0 - 7.9: easily understood by people who have completed a Ph.D. program or equivalent
    """
    # Tokenize sentences
    sentences = tokenize_sentences(text)
    sentences = [s for s in sentences if len(tokenize_words(s)) >= 2]
    num_sentences = len(sentences)

    # Tokenize words
    words = []
    for sentence in sentences:
        words.extend(tokenize_words(sentence))
    num_words = len(words)

    # Compute ARI
    char_count = sum([len(word) for word in words])
    avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0
    avg_word_length = char_count / num_words if num_words > 0 else 0
    ari_score = 4.71 * avg_word_length + 0.5 * avg_sentence_length - 21.43

    # Check for negative ARI score
    if ari_score < 0:
        return 0

    return round(ari_score, 1)

def composite_index(text, verbose = 0):
    """
    Output: a higher value means the text is simpler, and a lower value means the text is more complex

    """
    # NORMALIZE scores to 0-1 scale

    # Flesch - The score ranges from 0 to 100, with higher scores indicating easier readability. 
    # To normalize, we divide the score by 100.
    flesch = flesch_reading_ease(text) / 100
    
    # Dale - The raw score ranges from around 4.9 to 11.5. 
    # To normalize, we subtract a reasonable lower-bound value (4.0) and then divide by the maximum adjusted range (11.5 - 4.0 = 7.5).
    dale = 1 - (dale_chall(text) - 4) / 7.5

    # ARI - The score can range from negative values to positive values (e.g., -6 to 12). 
    # To normalize, we first add 6 to the score, and then divide by the maximum adjusted range (12 + 6 = 18).
    ari_score = 1 - (ari(text)) / 18  # Normalizing ARI assuming a minimum of -6 and a maximum of 12 for the range

    # INDEX WEIGHTING 
    # If our audience is primarily non-native English speakers, we might want to give more weight to the Dale-Chall score, which specifically accounts for difficult words.
    # For highly technical texts, you might want to prioritize the ARI or Flesch score, as it accounts for the number of characters per word, which can be a good indicator of technical jargon.

    WEIGHTS = (1/3, 1/3, 1/3)

    index = (WEIGHTS[0]) * flesch + (WEIGHTS[1]) * dale + (WEIGHTS[2]) * ari_score

    if verbose == 0:
        return index
    
    if verbose == 1:
        return flesch, dale, ari_score, index, WEIGHTS

# ================================================================================================================================================================================================
# TESTING
# The following are article stubs to test the effectiveness of the simplicity scorer

## if main 
if __name__ == "__main__":
        

    # SAMPLE DOCS
    stub_wikiEcons = ("Economics (wikipedia)", """
    The publication of Adam Smith's The Wealth of Nations in 1776, has been described as "the effective birth of economics as a separate discipline."52 The book identified land, labour, and capital as the three factors of production and the major contributors to a nation's wealth, as distinct from the physiocratic idea that only agriculture was productive. Smith discusses potential benefits of specialization by division of labour, including increased labour productivity and gains from trade, whether between town and country or across countries.53 His "theorem" that "the division of labor is limited by the extent of the market" has been described as the "core of a theory of the functions of firm and industry" and a "fundamental principle of economic organization."54 To Smith has also been ascribed "the most important substantive proposition in all of economics" and foundation of resource-allocation theory – that, under competition, resource owners (of labour, land, and capital) seek their most profitable uses, resulting in an equal rate of return for all uses in equilibrium (adjusted for apparent differences arising from such factors as training and unemployment).In an argument that includes "one of the most famous passages in all economics,"56 Smith represents every individual as trying to employ any capital they might command for their own advantage, not that of the society,a and for the sake of profit, which is necessary at some level for employing capital in domestic industry, and positively related to the value of produce.58 In this: He generally, indeed, neither intends to promote the public interest, nor knows how much he is promoting it. By preferring the support of domestic to that of foreign industry, he intends only his own security; and by directing that industry in such a manner as its produce may be of the greatest value, he intends only his own gain, and he is in this, as in many other cases, led by an invisible hand to promote an end which was no part of his intention. Nor is it always the worse for the society that it was no part of it. By pursuing his own interest he frequently promotes that of the society more effectually than when he really intends to promote it.The Rev. Thomas Robert Malthus (1798) used the concept of diminishing returns to explain low living standards. Human population, he argued, tended to increase geometrically, outstripping the production of food, which increased arithmetically. The force of a rapidly growing population against a limited amount of land meant diminishing returns to labour. The result, he claimed, was chronically low wages, which prevented the standard of living for most of the population from rising above the subsistence level.60 Economist Julian Lincoln Simon has criticized Malthus's conclusions.""")

    stub_wikiSpiceGirls = ("SpiceGirls (wikipedia)", """
    The Spice Girls are a British girl group formed in 1994, consisting of Melanie Brown, also known as Mel B ("Scary Spice"); Melanie Chisholm, or Melanie C ("Sporty Spice"); Emma Bunton ("Baby Spice"); Geri Halliwell ("Ginger Spice"); and Victoria Beckham ("Posh Spice"). With their "girl power" mantra, they redefined the girl-group concept by targeting a young female fanbase.1 2 They led the teen pop resurgence of the 1990s, were a major part of the Cool Britannia era, and became pop culture icons of the decade.The Spice Girls formed through auditions held by Bob and Chris Herbert, who wanted to create a girl group to compete with the British boy bands popular at the time. The Spice Girls quickly left the managers and took creative control over their sound and image. They signed to Virgin Records and released their debut single "Wannabe" in 1996, which reached number one on the charts of 37 countries.6 7 Their debut album, Spice (1996), sold more than 23 million copies worldwide,8 becoming the best-selling album by a female group in history.9 It produced three more number-one singles: "Say You'll Be There", "2 Become 1" and "Who Do You Think You Are"/"Mama". The second Spice Girls album, Spiceworld (1997), sold more than 14 million copies worldwide.10 They achieved three more number-one singles with "Spice Up Your Life", "Too Much" and "Viva Forever". Both albums encapsulated the group's dance-pop style and message of female empowerment, with vocal and songwriting contributions shared equally by the members. In 1997, the Spice Girls made their live debut concert tour and starred in a film, Spice World, to commercial success. In early 1998 the group embarked on their Spiceworld Tour, which was attended by an estimated 2.1 million people worldwide, becoming the highest-grossing concert tour by a female group, grossing an estimated $220–250 million in ticket sales.11 Halliwell left the Spice Girls mid-tour in May 1998. Following a number-one single, "Goodbye" in 1998, and a successful 1999 concert tour, the Spice Girls released their R&B-influenced third album, Forever (2000), which featured their ninth number one, "Holler"/"Let Love Lead the Way". At the end of 2000, the Spice Girls entered a hiatus to concentrate on their solo careers. They reunited for two concert tours, the Return of the Spice Girls (2007–2008) with Halliwell, and Spice World – 2019 tour, both of which won the Billboard Live Music Award for highest-grossing engagements, making the Spice Girls the top touring all-female group from 1998 to 2020. The Spice Girls have sold 100 million records worldwide,13 14 15 making them the best-selling girl group of all time,16 17 18 one of the bestselling artists, and the most successful British pop act since the Beatles.19 20 21 They received five Brit Awards, three American Music Awards, four Billboard Music Awards, three MTV Europe Music Awards and one MTV Video Music Award. In 2000, they became the youngest recipients of the Brit Award for Outstanding Contribution to Music. According to Rolling Stone journalist and biographer David Sinclair, they were the most widely recognised group since the Beatles.22 Other measures of the Spice Girls' success include iconic symbolism such as Halliwell's Union Jack dress, and their nicknames, which were given to them by the British press. Under the guidance of their mentor and manager Simon Fuller, their endorsement deals and merchandise made them one of most successful marketing engines ever, with a global gross income estimated at $500–800 million by May 1998.23 nb 1 According to the Music Week writer Paul Gorman, their media exposure helped usher in an era of celebrity obsession in pop culture.
    """)

    stub_wikiAstroPhysics = ("Astrophysics (wikipedia)", """
    Astrophysics is a science that employs the methods and principles of physics and chemistry in the study of astronomical objects and phenomena.[1][2] As one of the founders of the discipline, James Keeler, said, Astrophysics "seeks to ascertain the nature of the heavenly bodies, rather than their positions or motions in space–what they are, rather than where they are." Among the subjects studied are the Sun, other stars, galaxies, extrasolar planets, the interstellar medium and the cosmic microwave background. Emissions from these objects are examined across all parts of the electromagnetic spectrum, and the properties examined include luminosity, density, temperature, and chemical composition. Because astrophysics is a very broad subject, astrophysicists apply concepts and methods from many disciplines of physics, including classical mechanics, electromagnetism, statistical mechanics, thermodynamics, quantum mechanics, relativity, nuclear and particle physics, and atomic and molecular physics.

    In practice, modern astronomical research often involves a substantial amount of work in the realms of theoretical and observational physics. Some areas of study for astrophysicists include their attempts to determine the properties of dark matter, dark energy, black holes, and other celestial bodies; and the origin and ultimate fate of the universe.[4] Topics also studied by theoretical astrophysicists include Solar System formation and evolution; stellar dynamics and evolution; galaxy formation and evolution; magnetohydrodynamics; large-scale structure of matter in the universe; origin of cosmic rays; general relativity, special relativity, quantum and physical cosmology, including string cosmology and astroparticle physics.
    """)

    stub_bbc_entertainment = ("BBCEntertainmentPiece",
                            'The question of whether the campaign broke Academy guidelines is believed to have come down to a few specific posts that not only championed Riseborough but also made reference to her competitors - which is forbidden. One since-deleted Instagram post that has been under the spotlight was published by the official To Leslie account. It quoted Richard Roeper of the Chicago Sun-Times, who wrote: "As much as I admired [Cate] Blanchetts work in Tar, my favourite performance by a woman this year was delivered by the chameleonlike Andrea Riseborough." While there was no wrongdoing on the critics part for expressing his opinion, its possible that the To Leslie campaign could have got in trouble for choosing a quote that contrasted Riseborough with Blanchett.')

    stub_straitsTimes_business = ("STMoneyAndBiz",
                                "Mr Gautam Adani pulled off a crucial US$2.5 billion (S$3.3 billion) equity sale for his flagship company, largely thanks to existing shareholders, earning the Indian billionaire some reprieve after his empire was rocked by fraud allegations by short-seller Hindenburg Research. A failure to meet the fund-raising goal would have been a major blow to Mr Adani’s prestige and would have heightened concerns about the conglomerate’s debt load. The offering by Adani Enterprises was India’s largest follow-on share sale, and was fully subscribed on the final day, aided by a last-minute surge in demand from institutional investors. Interest from retail investors – whom Mr Adani was hoping to attract – was notably weak. While the share sale’s completion is a victory for Mr Adani after Hindenburg’s allegations put the offering in doubt, that is unlikely to fully dispel investor concerns about the conglomerate’s corporate governance. Apart from existing Adani shareholder Abu Dhabi’s International Holding Co (IHC), which accounted for 16 per cent of the purchases in the offering, anchor investors like Life Insurance Corp of India and an arm of Goldman Sachs Group also ploughed money in.")

    stub_straitsTimes_life = ("ST_Life" , """As a photographer, Chua Soo Bin is not one to state the obvious. This is clear even in the early photographs he took in the 1950s and 1960s. One particular photograph of the Bukit Ho Swee fire in 1961 shows only five remaining charred and smoking coconut palms. There was no need to show the devastated villagers who lost their homes. There is pathos without the drama.The photograph is included in the recently launched biography of the photographer, gallerist, art dealer and patron of the arts, who turned 90 in December.""")

    stub_maryLamb = ("MaryHadALitteLamb" ,
                    "Mary had a little lamb, Its fleece was white as snow. And everywhere that Mary went, The lamb was sure to go. He followed her to school one day, That was against the rule. It made the children laugh and play To see a lamb at school. And so the teacher turned him out, But still he lingered near, And waited patiently about Till Mary did appear. And then he ran to her, and laid His head upon her arm, As if he said ‘I’m not afraid, You’ll keep me from all harm.’ ‘What makes the lamb love Mary so?’ The eager children smile. ‘Oh, Mary loves the lamb, you know,’ The teacher did reply. ‘And you each gentle animal In confidence may bind, And make them follow at your call, If you are always kind.’")

    stub_appleNews = ("AppleLabourDispute", """
    Apple violated US labor laws through various workplace rules and statements made by executives, National Labor Relations Board officials determined after reviewing allegations from two former employees. 
    An NLRB official will file a formal complaint against Apple unless the company reaches a settlement with the former employees, who filed complaints about Apple's focus on secrecy. 
    An NLRB spokesperson confirmed to Ars today that the labor board's regional office "found merit to four charges alleging that various work rules, handbook rules, and confidentiality rules at Apple violated Section 8(a)(1) of the National Labor Relations Act because they reasonably tend to interfere with, restrain, or coerce employees in the exercise of their right to protected concerted activity." The regional office additionally "found merit to a charge alleging statements and conduct by Apple—including high-level executives—also violated the National Labor Relations Act," the NLRB statement said. That's apparently a reference to an email in which Apple CEO Tim Cook warned staff not to leak confidential information. As The New York Times wrote, the NLRB findings were in response to "five charges brought in late 2021 by two former Apple employees, Ashley Gjovik, an engineering program manager at Apple for six years, and Cher Scarlett, an engineer on the company's security team... Both women were involved in the activist group called #AppleToo that was collecting accounts of abuse, harassment and retaliation at the company." The ex-employees "accused the company of trying to prevent the group from collecting wage data from employees, including through harassment," and "said that the company's work rules prevented them from discussing wages, hours and conditions of employment," according to the NYT story. Gjovik's complaints alleged that "various Apple rules, including those relating to confidentiality and surveillance policies, deter employees from discussing issues such as pay equity and sex discrimination with each other and the media," according to Reuters. "Gjovik also cited a 2021 email from Apple Chief Executive Tim Cook that allegedly sought to stop workers from speaking to the press and said 'people who leak confidential information do not belong here.'" We contacted Apple about the NLRB finding and will update this article if the company provides a response. The regional office's finding that the charges have merit isn't an NLRB ruling but could lead to a formal charge against Apple. The NLRB statement said that "if the parties don't settle, the Regional Director will issue a complaint, prosecuting this charge in a hearing with an Administrative Law Judge, who could order remedies." An administrative law judge's decision could be appealed to the board, and a board decision, in turn, could be appealed to a federal appeals court. The NLRB's standard process calls for the agency to help the parties reach a deal. "When the NLRB investigation finds sufficient evidence to support the charge, every effort is made to facilitate a settlement between the parties. If no settlement is reached in a meritorious case, the agency issues a complaint," an NLRB webpage explains.""")

    stub_GQFeatures = ("GQFeatureOnArtist", """On 7 October 2019, Yves McCrae, a 29-year-old visual effects artist, received an email from a recruiter asking if he would be interested in a job at the Vancouver branch of the Moving Picture Company (MPC). Since its founding in London in 1970, MPC has grown to become one of the most prestigious and storied visual effects houses in the world, winning three Academy Awards for its CGI work on the films 1917, The Jungle Book, and Life of Pi. McCrae expected a rigorous recruitment process, involving multiple interviews and probing questions about his showreel and experience. The recruiter explained, however, there would be no interview. If he wanted the job, McCrae just needed to forward a copy of his passport and show up at the office the following week. “Is this legit?” he replied. Seven days later, McCrae arrived at MPC’s Vancouver studio, a red-brick building situated in the expensive, historic part of the city, less than a mile from the Art Institute of Vancouver, where he had studied visual effects (VFX) several years earlier. McCrae, who has a youthful face and auburn hair, half expected to have been the victim of a prank. Instead, he found the lobby crammed with dozens of other young VFX artists, some of whom he recognised from his work on other projects: Godzilla, Black Panther, Stranger Things. “The waiting room was literally packed,” McCrae recalls. “There was nowhere to sit.” While he waited, McCrae struck up a conversation with the artist he was squeezed up against. “Neither of us had ever seen anything like it before.” The group was then led into a screening room, a cinema-like auditorium with comfy chairs and insulated walls, where staff would watch the “dailies” – VFX-heavy shots that had been completed – throughout the day. One of the heads of the department walked in, McCrae recalled, and addressed the crowd. “We’ll get right into it. We have two projects you’re going to be working on: either Cats or Sonic.” McCrae felt the energy in the room perceptibly shift. Both films were already notorious, and not just among VFX artists. A few months earlier, Paramount Pictures had released the first trailer for Sonic the Hedgehog, a live-action adaptation of the Sega video game. Unlike his streamlined design of the games, film-Sonic embodied an eccentric quasi-realism, featuring a chattering row of teeth, beady, fearful eyes, and a questing snout. On social media, the design was immediately pilloried. A journalist from The Guardian described Sonic’s design as looking like “a cheap knock-off… toy your child might win at a fairground stand and then be terrified of.” To fans this was, he added, like a “200mph slap in the face.” """)

    stub_wiredeview = ("WiredReviewOnProduct", """For years now I have used the Garmin 245, which falls in the middle of Garmin's Forerunner line and is aimed mainly at runners. With the release of the Forerunner 255, Garmin has retired the 245. As with any fitness tracker, how much any of this will benefit you depends on what you're doing. For reference, my workout routine is body-weight-based, with alternating walking and sprinting sessions thrown in throughout the week. I also used the 255 on hikes, paddleboarding, and for sleep tracking. Like nearly all non-touch Garmin watches, the Forerunner 255 has five buttons, three on the left side and two on the right. I find the buttons more reliable to navigate with than the touchscreen models, but the main thing to note is there's no touchscreen here. The watch face is fully customizable, with a good selection of default faces you can use to customize to your liking. There are quite a few new features worth mentioning, but the one I was most excited about is the sleep tracking. The Forerunner 255 tracks heart-rate variability (HRV) and sleep stages and gives you an overall Sleep Score, with a new Morning Report, which includes the company’s Body Battery feature as well as a daily greeting, the weather, and other tidbits. It's similar to what Apple offers. To test the accuracy of the Forerunner 255, I had my wife, who uses an Oura ring to track sleep, use it for a few weeks. (The Forerunner requires 19 days of use before it will start making recommendations based on the sleep data it has collected.) To keep this review to a reasonable length, the result was that she is no longer worried about replacing her Oura, given that company’s disappointing trajectory. The Forerunner's data mostly matched that of the Oura and is very nearly as comprehensive. But it doesn't track body temperature while sleeping and is missing some other features the Oura offers. I particularly liked my Morning Report, a good way to get a quick overview of where you're at and what you want to do that day, training-wise. The other thing that jumped out at me in the first week was the new auto-detection features. Once I started the Endurance Training activity, the Forerunner knew when I was doing push-ups and logged them. It knew when I was walking, running, and resting. It's particularly well-suited to interval training thanks to the automatic detection. Also useful are the training and recovery tools, which previous models I've used lack. They're not as comprehensive as what you'll find in the more expensive Forerunner 955, but together with the sleep tracking they can give you a more accurate impression of where you're at and how to get to where you want to be. That said, fitness tracking remains an inexact science. The Forerunner 255 is convinced I climb seven flights of stairs a day on average. I assure you, I do not. Other features worth mentioning include multiband GPS support (with dual-frequency support), which makes for more accurate GPS tracking, a barometric altimeter and compass (useful for hitting the trail), and a ton more cycling support, including VO2 max and the ability to connect to many bike gadgets via Bluetooth. The GPS support is worth noting because it is much more accurate, but it takes a heavy toll on the battery. Battery life on the 255 is quite good overall. In fact, it's good enough that I didn't pay much attention to it. Using it the way I do, with an activity on for about an hour a day, GPS tracking on, but Pulse Ox set to on-demand only, I got about five days. The longer you use an activity the more battery it'll take. While I had the watch, my wife did a 12-hour walk, so naturally I gave it to her to test. When she got back, the battery had run down to about 40 percent, which would put it about 30 hours of continuous use.  I find the real key to long battery life is to turn off continuous pulse-ox measurements and other battery-intensive features when you don't need them. The final thing worth mentioning is that you can now fully configure your watch using the Garmin app on your phone. Naturally you can change settings from the watch, but the ability to set things up on your phone makes life so much easier, especially given the emphasis here on creating workout plans through Garmin Coach based on future events. The daily suggested workouts can be connected to calendar events, like an upcoming race. The app can automatically create workouts for that distance.""")

    texts = [
        stub_wikiEcons,
        stub_wikiAstroPhysics,
        stub_wikiSpiceGirls,
        stub_bbc_entertainment,
        stub_straitsTimes_business,
        stub_straitsTimes_life,
        stub_maryLamb,
        stub_appleNews,
        stub_GQFeatures,
        stub_wiredeview
    ]

    # SCORING
    df = pd.DataFrame(columns=["article", "norm_flesh", "norm_dalechall", "norm_ari", "index", "WEIGHTS"])
    for text in texts: 
        print("===================")
        print(text[0])
        print("Flesch Reading Ease:", flesch_reading_ease(text[1]))
        print("Dale-Chall:", dale_chall(text[1]))
        print("ARI:", ari(text[1]))
        print("Composite Index:", composite_index(text[1]))
        flesch, dale, ari_score, index, WEIGHTS = composite_index((text[1]), verbose = 1)
        temp_data = pd.DataFrame({
            "article": [text[0]],
            "norm_flesh": [flesch],
            "norm_dalechall": [dale],
            "norm_ari": [ari_score],
            "index": [index],
            "WEIGHTS": [WEIGHTS],
        })

        df = pd.concat([df, temp_data], ignore_index=True)

    # PLOTTING 
    # Sort the DataFrame by index
    df = df.sort_values('index', ascending= False)

    # Create a bar plot for the index
    barplot = sns.barplot(data=df, 
                        x='article', 
                        y='index', 
                        color='black') 

    # Add the normalized scores as line plots
    for metric, color in zip(['norm_flesh', 'norm_dalechall', 'norm_ari'], ['darkblue', 'blue', 'lightblue']): 
        plt.plot(df['article'], 
                df[metric], 
                marker='o', 
                color=color, 
                linestyle='--', 
                label=metric, 
                alpha=0.7) # update to make lines dotted and 0.7 opaque

    # Add a title to the plot
    plt.title(f'Sample Scoring \nWeights: {df.iloc[0]["WEIGHTS"]}')
    plt.ylabel('Simplicity Index')
    plt.xlabel("Sample Articles")
    plt.legend()
    plt.xticks(rotation=90)
    plt.show()
