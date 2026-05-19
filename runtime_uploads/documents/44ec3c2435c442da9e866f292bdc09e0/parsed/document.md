GenRole: Personalizing Role Play for Educators Supporting Autistic Students’ Social Interaction Learning

Yixuan Li

Keyi Zeng

Jiaqi Zong

Computational Media and Arts Thrust, Information Hub The Hong Kong University of Science and Technology (Guangzhou) Guangzhou, China Tsinghua University Shenzhen, China yixuan-li25@mails.tsinghua.edu.cn

Computational Media and Arts Thrust, Information Hub The Hong Kong University of Science and Technology (Guangzhou) Guangzhou, China Tsinghua University Shenzhen, China zky23@mails.tsinghua.edu.cn

The Hong Kong University of Science and Technology (Guangzhou) Computational Media and Arts Thrust, Information Hub, China ETH Zürich Zürich, Switzerland jizong@ethz.ch

Yingying Zhang

Hongzhu Deng

Li Wang

The Third Affiliated Hospital ,Sun Yat-Sen University Guangzhou, China zhangyy335@mail2.sysu.edu.cn

The Third Affiliated Hospital, Sun Yat-Sen University Guangzhou, China denghzh@mail.sysu.edu.cn

Nansha Qihui School Guangzhou, China nanshaqihui@outlook.com

Xin Tong ∗

The Hong Kong University of Science and Technology (Guangzhou) Guangzhou, China The Hong Kong University of Science and Technology Hong Kong, China xint@hkust-gz.edu.cn

Figure 1: GenRole is a system that incorporates progressive role play and enables teachers to create personalized components based on the varied needs of autistic children, aiming to enhance their social skills.

∗ Corresponding Author.

This work is licensed under a Creative Commons Attribution-NonCommercialNoDerivatives 4.0 International License. CHI ’26, Barcelona, Spain

Abstract

Role-play is widely used to empower autistic children to explore social interaction and dynamics on their own terms, navigating

© 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2278-3/26/04 https://doi.org/10.1145/3772318.3791948

CHI ’26, April 13–17, 2026, Barcelona, Spain

neurotypical social conventions to shape social expressions in ways that align with their own traits and needs, fostering a stronger sense of agency. However, existing approaches typically rely on fixed content, requiring educators to design materials, which creates a significant burden on manual preparation. According to insights from a formative study, we developed GenRole, a generative AI system that enables educators to design personalized role play class activities. GenRole supports a progression from simple to complex interactions and allows for personalization of characters, settings, and dialogues that meet the needs of autistic learners. We conducted a pilot study with 16 educators, followed by a two-week evaluation study with 11 autistic children and their teachers. Results show that GenRole enhances the efficiency and flexibility of role play design while improving instructional support, offering design insights for creating personalized components that help educators deliver more engaging and individualized social interaction learning for autistic children.

CCS Concepts

• Human-centered computing → Accessibility systems and tools.

Keywords

Generative AI, Personalization, Social skill training, Autistic children, Role play

ACM Reference Format: Yixuan Li, Keyi Zeng, Jiaqi Zong, Yingying Zhang, Hongzhu Deng, Li Wang, and Xin Tong. 2026. GenRole: Personalizing Role Play for Educators Supporting Autistic Students’ Social Interaction Learning. In Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems (CHI ’26), April 13–17, 2026, Barcelona, Spain. ACM, New York, NY, USA, 19 pages. https://doi.org/10.1145/3772318.3791948

1

Introduction

Autism spectrum disorder (ASD) is a neurodevelopmental condition or neurotype that differs from the majority population in cognitive and sensory processing [4]. When autistic children 1 are in neurotypical-dominated environments, it can lead to misunderstandings in forming social relationships and adapting to everyday social situations [66]. These misunderstandings include interpreting non-verbal cues [96], experiences of loneliness [8], and so on. However, since autistic individuals have diverse perspectives on neurotypicality [4], it is critical to align intervention and education goals with each individual’s views and preferences. Some individuals may aim to reduce autistic behaviors, while others may prefer to focus on building skills and learning compensatory strategies. In either case, the most meaningful goals are those that lead directly to improved quality-of-life outcomes [4]. Research shows that autistic children’s social understanding and well-being can be empowering when educators 2 advocate for environmental changes that respect and support autistic differences

1 Our work uses identity-first language, referring to the population as “autistic children”,

in line with the preferences of the broader autism community and recent academic advocacy [16] 2 For this study, we use the term educators to refer to the teachers and specialists who worked directly with autistic children.

Li et al.

[4]. Educators also employ thoughtfully designed instructional approaches to foster mutual understanding and inclusive communication, while empowering autistic students to better navigate neurotypical social rules and skills [33, 37, 39, 76, 97], such as social stories [37], power cards [39], comic strip conversations [16], and which are effectively delivered via video modeling [33] and role play [52]. Among these approaches, role play has been widely applied as an effective way to support social interaction learning among autistic children [22]. Rather than being treated as a tool for correcting behavior, role-play can be understood as a co-creative process through which educators and autistic children jointly develop a mutual understanding of social interaction. By providing immersive learning experiences [22] and enabling children to take on specific roles in various scenarios [92], it serves as a supportive activity. Role play also supports emotional expression through facial expressions and body language [30], offering potential benefits for the development of social interaction skills in autistic learners. Rather than teaching them to “act typical,” we aim to empower autistic children to explore social dynamics on their own terms, building agency rather than compliance. Previous HCI research has explored how digital role play applications can create rich social interactions and effectively engage students [25]. This has found widespread practical application, particularly in the field of accessibility [59], including approaches to supporting autistic children’s understanding of neurotypical social skills [58]. Nevertheless, since autistic children exhibit variability in preferences [65] and literacy levels [25], educators face challenges in designing appropriate role play scenarios, narratives, and characters, as well as in selecting an instructional approach that suits each child. Given these challenges, educators need to develop personalized role play components that align with each child’s specific needs, proving effective in enhancing communication and social interaction [58], thereby supporting their understanding [69]. However, implementing this personalization remains challenging for educators. Nowadays, many role play approaches still rely on fixed content [13, 53], which cannot address individual needs. Additionally, personalizing such role-play activities requires educators to develop diverse learning materials [29], leading to increased resource demands [26]. As a result, educators often struggle to create varied, engaging, and developmentally appropriate social stories and role-play scenarios tailored to each child’s learning goals. AIassisted tools can help address this challenge by automating and personalizing learning content [7]. They also offer scalable and adaptable solutions by generating social stories [78], images [101], and storylines based on selected characters [32], thereby meeting the diverse needs of autistic children. Therefore, we propose the following research question: RQ1: What are educators’ perceptions and needs regarding personalized role play materials for supporting autistic children’s social interaction learning? RQ2: How do educators use and perceive a GenAI-powered tool that creates personalized role play components for autistic children’s social interaction learning? RQ3: What challenges and opportunities arise in implementing AI-powered role play activities for autistic children? Therefore, we present GenRole, a prototype that supports progressive role play activities and helps educators design personalized

GenRole: Personalizing Role Play for Educators Supporting Autistic Students’ Social Interaction Learning

components tailored to autistic children’s diverse needs. These components include social skill targets, visual elements (e.g., characters, background scenes, and reinforcers), and role-play scripts, all designed to scaffold the development of social interaction skills in autistic children. Through a formative study with 3 educators, we identified limitations of in-class role play methods and explored ways to design personalized components tailored to children’s needs. Based on these insights, we developed the GenRole prototype and then conducted a pilot study with 16 educators of autistic children to iterate on the design. Finally, we conducted a user study involving 11 teacher-student pairs to evaluate its usability and effectiveness. Our findings suggest that GenRole’s role play feature effectively enhances mutual understanding in social skills, and the personalized components in role play can adapt to children’s learning processes, improving children’s social skills. Our main contributions include: (1) GenRole, a GenAI-powered tool that enables educators to personalize role play content for supporting autistic students’ social interaction learning. (2) Empirical findings and guidance on creating personalized role play components, from a formative study, a pilot study with educators, and a main user study involving educator–student pairs; (3) Design implications on how AI-assisted role play and other interactive education systems can help educators deliver individualized social learning experiences for autistic students.

2 Related Works 2.1 Social Skills Education Approaches for Autistic Children

Social skills are essential for helping autistic children navigate various social situations [23]. Prior studies have explored various aspects of social skills understanding and learning for autistic children, including emotion recognition [14, 101], neurotypical-based social norms [38, 114], identifying emotions or intentions [82, 94], conversation, gesturing, socially appropriate behaviors [82], perspective taking [94], cooperation [31, 44, 56], empathy and self-control [44, 56]. However, autistic social interactions are often misframed as a "lack of social insight" [77, 110], particularly in areas such as friendship formation, social connection building, and quality of life maintenance. The double empathy problem [83] challenges this deficit view, highlighting that communication barriers between autistic and neurotypical individuals stem from mutual differences in cognition and perception, not a one-sided impairment. As these differences become more pronounced during adolescence [9], social skills education should avoid pressuring autistic children to conform to neurotypical norms. Instead, it should focus on fostering mutual understanding of autistic individuals in navigating neurotypical contexts while respecting their neurodiverse traits. Several educational approaches, such as Social Skills Training (SST) [101], Expressive Therapies Continuum (ETC) [74], and Cognitive Behavioral Therapy (CBT) [47], and tools like Social Stories™ [43], provide a foundation for developing social skills to help children navigate social situations. However, these approaches typically rely on the listener’s imagination to process the content [17, 43]. Unlike traditional, non-interactive methods, role play is a participatory approach that prioritizes autistic individuals’ preferences [62],

CHI ’26, April 13–17, 2026, Barcelona, Spain

linking social behaviors with cognitive and emotional understanding [46] and simulating real-world scenarios to foster experiential learning [68, 79, 107]. Role play has also been shown to support autistic children in navigating neurotypical social conventions (e.g., eye contact, sportsmanship such as giving compliments, and conversational turn-taking and maintenance) [81]. This approach provides opportunities for autistic children to explore how social cues operate in specific contexts [58] and to shape social expressions in ways that align with their own traits and needs.

2.2

Challenges in Social Interaction Instruction for Educators

For educators, teaching social interaction is essential to foster autistic children’s independence and peer engagement [73]. However, they face significant challenges in delivering this instruction and have expressed concerns regarding their insufficient knowledge and skills training on Autism [1]. Although growing awareness is attracting more educators to the field, many feel underprepared when designing instruction tailored to autistic students’ specific interests and needs [1]. Educators report difficulty in incorporating students’ intense interests into lessons and responding appropriately to behavioral adaptation needs in classroom contexts [108]. Preparing individualized social learning content is time-intensive, and educators often operate under high workload pressures and competing academic demands [1]. Structural barriers such as a lack of appropriate resources, including assistive technology and software, further complicate instruction [73, 108]. Effective social interaction learning, especially through role play, requires materials aligned with each child’s cognitive level, interests, and behavioral objectives [25, 65]. Yet, existing materials often rely on fixed narratives and rigid structures [13, 53], which lack flexibility and contextual relevance. Moreover, aligning activities with students’ Individualized Education Programs (IEPs) and communication profiles necessitates a degree of personalization that is difficult to achieve at scale [26]. In this context, AI-assisted tools are emerging as a promising way to provide educators with case-based learning and scalable resources [20], enabling them to create personalized content more efficiently [36]. Therefore, we present GenRole, a tool that helps teachers generate personalized role-play scripts and visual materials tailored to specific learning contexts.

2.3

Approaches to Using Generative AI for Personalized Storytelling

Autistic children exhibit significant diversity in cognitive abilities, social preferences, interests, and sensory processing preferences [27, 78]. Previous studies have shown that non-personalized approaches, such as fixed stories or flashcards, often result in low engagement and limited generalization [74, 78]. Existing HCI systems offer important forms of flexibility but still rely on predefined content and manual authoring. For example, the Authorable Virtual Peer (AVP) framework allows children to directly create or script interactions with virtual peers when they are able and interested, which may demand higher levels of skill and ability from users [102], StarRescue structures social learning around a predefined turn-taking mechanism, limiting adaptive customization [10], and Lend-a-Hand integrates core social teaching into an AI-assisted

CHI ’26, April 13–17, 2026, Barcelona, Spain

gesture-recognition pipeline, but faces challenges in adapting to individual progress and needs [60]. Similarly, in the domain of role play based approaches, Ke et al. developed a desktop VR learning environment to support socialoriented role play for autistic children [63], and Lee et al. combined augmented reality with physical elements in tabletop role play games to teach social interaction skills [69]. However, the storylines, scenes, and scripts in these systems are largely predefined, involve extensive preparation, and offer limited flexibility for further personalization. Taken together, personalization helps bridge life experiences with stories[101], strengthen outcomes[74], and that autistic children are more likely to engage with stories that match their interests and everyday contexts [78]. In recent years, generative AI tools have demonstrated significant potential in supporting children’s personalized learning and creative expression [49, 75, 101]. These tools simplify the creation of personalized components, enabling content to be produced more efficiently [36]. For example, AIStory allows children to customize visual stories by selecting stickers for characters and scenes [49], StoryDrawer enables collaborative drawing and story completion based on children’s oral narratives [113], and systems like ID.8 provide auto-generated story scenes and scripts to make the creative process more efficient [6]. Generative AI has also been used to support autistic children. Emoeden creates personalized conversations based on children’s preferences to foster emotional recognition and expression [101], Amy acts as a virtual conversational companion that delivers social stories tailored to autistic children’s needs [37]. While generative AI can improve efficiency, over-reliance on AIgenerated content may limit creativity (e.g., idea diversity) and reduce children’s autonomy [54], and may not provide reliable results [48], limiting children’s independence and exploration. Our work expands this area by providing more customizable components that support educator-facing personalization of social interaction activities. Teachers can also upload, add, and modify content to personalize the experience further.

3

System Design

This section presents a formative study (Fig. 2) with teachers of autistic children to identify challenges in role play based social skills teaching and explore opportunities for personalization with generative AI. Based on the formative study’s findings, we designed the GenRole, a system that incorporates progressive role play and allows teachers to create personalized components based on the diverse needs of autistic children. This section also details GenRole’s two main design features, user flow, and technical architecture.

3.1

Formative Study

We conducted semi-structured online interviews with three instructional supervisors 3 from different therapy centers (none of these participants joined later studies), all of whom had extensive experience overseeing classroom instruction in various settings, to gather insights that informed our design. Participants were recruited via networks and social media, with all three studies in GenRole being approved from Hong Kong University of Science and Technology

3 Instructional supervisors are responsible for observing lessons, providing teaching

feedback, and supporting teachers’ instructional improvement.

Li et al.

(Guangzhou) Institutional Review Board (IRB). All participants were compensated for their time. We aimed to: (1) Examining the use of role play in social interaction skills learning and identify the challenges that teachers and children encounter. (2) Exploring design opportunities to use generative AI to personalize role play. Each participant’s interview lasted approximately one hour and was recorded audibly. All three experts had extensive experience in inclusive social interaction communication for autistic children and a basic understanding of generative AI (Table 1). Based on the interviews, we summarized the role play practices (Fig.3). Teachers typically spent a week gathering materials and content to create personalized stories for autistic children, using slides to combine photos and text (P2, P3). After preparing the content, teachers guided children through dialogues and variation practice, continuously updating the content to reinforce children’s memory. After multiple rehearsals, teachers engage children in field practices to transfer skills to real-life situations (P2). Overall, teachers faced challenges that they relied on time-consuming content preparation and had limited variety. P3 highlighted that generative AI could ease teachers’ workload by simplifying content sourcing, generating personalized images and text that are tailored to a child’s learning progress. For the choice of generated visual style, P3 and P1 recommended using cartoon style, as young children tend to prefer cute and lively images, which can help alleviate their social anxiety [111]. Additionally, P2 suggested that the design could incorporate multiple practice modes from classic role play to meet autistic children’s learning needs. This would allow for repeated practices, potentially enhancing their ability to learn social interaction skills more effectively by providing varied scenarios and a pace suited to individual progress.

3.2

Design Features

Drawing from the key challenges and suggestions from the formative study, we identified the following design features.

3.2.1 Personalization Features. GenRole has three key personalization components: social skills, visual components (characters, scenes, and reinforcers), and scripts, as shown in Fig. 4(a). In a typical role play scene, two participants are involved: the participant (A), who plays themselves, and the teacher (C), who assumes the role of another person from the participant’s life (B) [79]. Based on this framework, GenRole defines two characters, multiple background scenes, and a reinforcer as its visual components. Educators first select a specific social skill as the learning goal, then generate personalized visual components that align with the child’s life experiences. Afterward, GenRole uses these components to generate corresponding role play scripts, providing a more tailored mutual understanding experience. This approach results in social outcomes that can be applied to real-world situations [81].

Personalized Social Skills. To personalize the role-play story scripts for various social skill needs, we reviewed existing scales [34, 99], along with how social skills have been categorized and trained in the literature. These include categories such as Social Rules [44, 56], Basic Communication [31], Social Emotion [82, 94], Problem Solving [94], Social Conversation [82], and Friendship Building [31, 44, 56]. Building on these categories, we adapted the content

GenRole: Personalizing Role Play for Educators Supporting Autistic Students’ Social Interaction Learning

CHI ’26, April 13–17, 2026, Barcelona, Spain

Figure 2: User study workflow in GenRole’s three studies.

Table 1: Demographics of participants in the formative study.

Participants

Gen

Age

Job Title

Experience with Autistic Children

P1 P2 P3

F F F

34 31 30

Instructional Supervisor, Behavioral Therapist Instructional Supervisor Instructional Supervisor, Speech-Language Therapist

Therapy Center (8 years) Therapy Center (10 year) Therapy Center (9 years)

Teacher Preparation

Teacher-Child Practice

Dialogue Practice

Content Preparation

Field Practice

Practice fixed dialogue for the scenarios

Videos + Picture + Dialogue Books

Design

Practice

Stories PPT

Variation Practice

Practice the same social skills in varied: Tasks + Scenarios

After multiple rehearsals

Experience real-life situations

Figure 3: Classic role play practices in the classroom.

to local cultural contexts by incorporating Chinese social norms, communication patterns, and everyday scenarios. We then selected a focused yet flexible subset of skills [52, 53, 106] suitable for roleplay story generation, emphasizing skills that can be expressed through narrative and dialogue, rather than those requiring extensive physical or behavioral interaction. More importantly, teachers can input new social skills tailored to autistic children’s needs. Detailed content for each category can be found in the supplementary material.

Personalized Visual Components. Previous role play for autistic children has often relied on existing printed cards to provide visual components [13, 53]. In GenRole, teachers can generate the visuals of two characters, multiple background scenes, and one reinforcer, all of which are key components of typical role play processes [100]. GenRole generates personalized visuals based on teachers’ input prompts (e.g., an aquarium with sharks). GenRole allows teachers to print and cut out the generated visuals to further engage children, creating “The Tangible Card”. These visuals can be used as tangible props or character cues as dynamic content [72]to enhance children’s engagement, and promote fine motor skills and body movement during role play [68].

Personalized to Role Play Script. After the teachers select a social skill and generate the corresponding visual components, GenRole generates a personalized role play script. These scripts are tailored to the child’s specific needs and selected social skills, offering a more engaging and interactive learning experience. The generated

script includes a dialogue page, tips on the Tips Page, and encouragement on the Reward Page after each scene, detailed in Section 3.3. A specific example of the generated content is shown in the supplementary material.

3.2.2 Progressive Role Play Modes for Social Interaction Skill Development. Inspired by progressive methods used in skill learning [41], we designed three role play modes to enhance children’s learning: Echo Mode, Character Mode, and Exploration Mode (Fig. 4).

(1) Echo Mode: Teachers read the role-play script aloud to guide children through the story. Students listen, follow along, and begin to recognize key social cues. This stage helps teachers model appropriate social behaviors while providing students with a structured introduction to the situation. (2) Dialogue Mode: Teachers present the dialogue without characters, prompting students to imagine who is speaking and what is happening. Students actively reason through the scenes, practice dialogue exchange, and explore the meaning of social interactions with teacher guidance. (3) Exploration Mode: Teachers provide only the social scenes, encouraging students to use creative thinking to construct characters, dialogues, scenes, and situations [88]. Students engage in creative role play, applying social interaction skills in new contexts while teachers observe, scaffold, and give feedback.

CHI ’26, April 13–17, 2026, Barcelona, Spain

Li et al.

Figure 4: Visual representation of components in Echo Mode, Dialogue Mode, and Exploration Mode

Figure 5: The user flow in GenRole consists of two stages: the personalization stage and the role play stage. In the personalization stage, (a) the user begins with the Introduction Page and User ID. Then, (b) they can view previous content through the Student Profile (History). (c) New users proceed to choose a social skill. Next, (d) the user provides a generative input description, followed by (e) the system generating visual components, (f) the role play script, and (g) the Exploration Mode script. In the role play stage, (h) users can choose different modes for role play (the new Character Mode being introduced in Section 4.3.2). During the role play, (i) the Dialogue Page displays dialogues between the teacher and child, and (j) the Tips Page offers strategic tips to guide the child in understanding subsequent interactions or tasks. Finally, (k) the Reward Page appears after each scene, showcasing the reinforcer and offering encouragement. The GenRole’s interfaces in the user study were originally presented in Chinese, but have been translated into English in this manuscript for clarity.

3.3

User Flow

GenRole provides two stages: the generation stage and the role play stage. In the generation stage, teachers can create role play content directly on the website (Fig. 5). (1) Personalization Page: After reading the introduction page, teachers can input their user ID to create a role play (Fig. 5 (a)). They can also view their student profile to access previous content (Fig. 5 (b)). During the generation stage, teachers can select or input

a new skill (Fig. 5 (c)) based on the child’s needs. They then generate visual components by providing details such as name, appearance, age, and reinforcement for character creation (Fig. 5 (d, e)). Once all visual components are generated or uploaded, teachers can generate the story script and manually edit it as needed (Fig. 5 (f, g)). (2) Role Play Stage: After completing the generation stage, all personalized role play content is automatically transferred to the role play stage. Children and teachers should begin with Echo Mode to understand the story. During the role play, the teacher can control

GenRole: Personalizing Role Play for Educators Supporting Autistic Students’ Social Interaction Learning

the dialogue playback. They can then freely choose other modes (Fig. 5 (h)) to continue the role play. During the role play, there are three types of interactive pages: the Dialogue, Tips, and Reward pages (Fig. 5 (i, j, k)):

(1) The Dialogue Page displays the conversation between the teacher and child. Each dialogue is designed to incorporate social interaction learning tasks within various social contexts. The teacher plays a crucial role in guiding the conversation by asking questions or keeping it going. (2) The Reward Page features a reinforcer designed to provide positive feedback. This element, with its generated image, appears after each scene to acknowledge and encourage the child’s efforts and achievements. (3) The Tips Page is strategically placed throughout the script to guide the child. It helps set up subsequent interactions or tasks, ensuring the child understands what is expected and how to achieve their goals.

3.4

Technical Architecture

Our technical architecture consists of two main parts: input and output (Fig. 6). Teachers first input the story and children’s personalized data through a Vue.js [112]-based interface. This data is stored in a MySQL [86] database and used to populate prompt templates. The system then transmits this data to a Flask server [87], which sends the prompts to the OpenAI API to generate personalized components. We utilize GPT-4 and DALL·E [84] to generate the role play script and visual components, including characters, backgrounds, and reinforcers. Once the scripts and characters are generated, they are reviewed and approved by teachers. After approval, a request is made to iFLYTEK [57] to synthesize the voice based on the characters’ gender and age. Once all personalized components are confirmed by the teachers, the scripts, visuals, and audio are sent to the Unity-based [103] application. The final output is presented as software, allowing teachers and children to begin the role play. GenRole produces four types of visuals: child and teacher characters, background scenes, and reinforcer icons. These are created with DALL·E 3 using fixed parameters, generating one image per request in 1024x1024 pixels (1792x1024 pixels for backgrounds). The workflow includes image creation, background removal for characters and icons, and storage in the GenRole database for reuse. Alternatively, users may upload photos and apply background removal via Remove.bg [91]. The prompts for visual generation consist of universal rules and user-defined inputs. Appendix.A provides an example along with the visuals generated by DALL·E 3. In crafting the prompt for the GenRole system, we primarily followed the C.O.S.T.A.R. framework, which stands for Context, Objective, Style, Tone, Audience, and Response Format [98]. Supplementary material illustrates how this framework applies to the prompt design for the GenRole system. Additionally, our prompt incorporates chain-of-thought reasoning [109] to require GPT-4 [85] to generate multiple related scenes that logically build on each other, promoting a coherent narrative flow. It also utilizes one-shot learning to enhance the model’s ability to generate relevant content with minimal input. To ensure system security and stability, we

CHI ’26, April 13–17, 2026, Barcelona, Spain

also left a manual editing window for the teacher to adjust the AI’s results at any time.

4

Pilot Study

To assess GenRole’s features and usability, we conducted a pilot study (Fig. 2) with 16 teachers, gathering feedback via interviews and questionnaires to identify drawbacks and gather suggestions. Hong Kong University of Science and Technology (Guangzhou) IRB approved this research study. All participants received compensation for their time in this study. Based on the findings from the pilot study, we iterated on the GenRole prototype, refining the role play modes, visual components, and language expression.

4.1

Participants

A total of 16 participants (all female, aged 19 to 47) from seven cities took the study (Table. 2). Participants had to (1) have over a year of experience in special education with autistic children and (2) own a computer to run GenRole.

4.2

Study Procedure

The pilot study lasted approximately 70 minutes on average for each participant. First, participants completed a 40-minute preparation session, which included a demographics survey and an introduction to GenRole. Participants then envisioned a specific child they had taught, taking into account the child’s learning process, challenges, and specific social interaction skill needs, and used GenRole to generate personalized content for their child. They also used GenRole to experience the three modes on their computers, simulating role play interactions with that child. Next, participants took part in a 30-minute semi-structured interview to provide feedback. We also collected feedback using a 5-point Likert scale on the System Usability Scale (SUS) [70], and a questionnaire about assessing participants’ overall satisfaction, the value of AI-generated stories, and a comparison to traditional role play. The interview outlines and questionnaire are shown in the supplementary material.

4.3

Findings and Iteration

We reported participants’ ratings on the SUS and questionnaire, and identified potential improvements in GenRole.

4.3.1 General Usability and Self-reported Ratings. Findings from the SUS scale yielded an average score of 73.42. Teachers’ overall satisfaction was generally positive (Mean = 3.95, SD = 0.85), indicating that GenRole is user-friendly and well-received by participants. In terms of generative AI support for social interaction skills teaching, participants provided positive feedback on the educational value of AI-generated stories (Mean = 4.16, SD = 0.69) and the comparison to classic role play methods (Mean = 3.89, SD = 0.78). Taken together, these results demonstrate that GenRole leverages generative AI to offer valuable support and serve as a meaningful alternative to classic role play methods. An exploratory comparison showed that participants with less than three years of experience rated the system more favorably than those with more than three years. Specifically, less-experienced participants gave higher ratings for the educational value of AIgenerated stories (𝑀 𝑙𝑒𝑠𝑠 = 4.44, 𝑆𝐷 𝑙 𝑒𝑠𝑠 = 0.53 vs. 𝑀 𝑚𝑜𝑟 𝑒 = 3.71,

CHI ’26, April 13–17, 2026, Barcelona, Spain

Li et al.

Figure 6: Technical architecture in GenRole.

Table 2: Demographics of participants in the pilot study.

Group

Participant

Gender

Age

Education

Experience

Teaching

Teacher

P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11 P12 P13 P14 P15 P16

F F F F F F F F F F F F F F F F

34 34 29 33 36 47 19 20 20 21 20 30 24 20 20 25

Postgraduate Postgraduate Postgraduate Bachelor Postgraduate Postgraduate Bachelor Bachelor Bachelor Bachelor Bachelor Bachelor Associate Degree Bachelor Bachelor Bachelor

13 years 14 years 5 years 5 years 10 years 21 years 1 year 1 year 1 year 2 years 1 year 9 years 3 years 1.5 years 1 year 2 years

Parent skills training Clinical medicine, contact Cognitive skills training courses Social skills for school students Social support Child behavior development Special education Special education Special education Special education Social support Child behavior development Rehabilitation therapy Special education Special education Special education

𝑆𝐷 more = 0.76), overall satisfaction (𝑀 𝑙 𝑒𝑠𝑠 = 4.56, 𝑆𝐷 𝑙 𝑒𝑠𝑠 = 0.53 vs. 𝑀 𝑚𝑜𝑟 𝑒 = 3.29, 𝑆𝐷 𝑚𝑜𝑟 𝑒 = 0.76), and comparison to classic role play methods (𝑀 𝑙 𝑒𝑠𝑠 = 4.33, 𝑆𝐷 𝑙 𝑒𝑠𝑠 = 0.50 vs. 𝑀 𝑚𝑜𝑟 𝑒 = 3.29, 𝑆𝐷 𝑚𝑜𝑟 𝑒 = 0.76). The average SUS score for less-experienced participants was also higher (78.33 vs. 65). A possible explanation is that moreexperienced participants may have developed strategies that mitigate challenges faced more intensely by junior educators, thereby reducing their perceived need for GenRole’s support. However, qualitative feedback was largely consistent across participants’ experience levels, suggesting shared perceptions of the system’s strengths and limitations.

4.3.2 GenRole’s Three Progressive Modes. Participants praised the structured arrangement of GenRole’s three modes, which supported repetition training and reinforced social interaction skills from progressive perspectives. Teachers noted that Echo Mode helped them fully understand the story, Dialogue Mode emphasized repeating scripted lines to improve verbal social interaction skills (P13), and

Exploratory Mode offered teachers more freedom to guide students. As P11 noted: “I think Mode 3 (Exploratory Mode) offers a lot of freedom...the scenario helps them adapt to different situations—something that might not be covered in a regular class.” However, the coherence between the modes needed improvement (P6). Some participants proposed adding a fourth mode by removing the script, allowing children to use their memory and apply learned social interaction skills (N=4), while helping teachers assess children’s ability to apply skills in new contexts. This fourth mode could help clarify concepts for students and enhance their memory (N=3). Therefore, we added a Character Mode between the Dialogue Mode and the Exploratory Mode. In Character Mode (Fig. 7), we removed the role play script while retaining the characters, scenes, and reinforcers. This modification allowed children to focus more on the visual cues provided by the characters to understand social situations, to encourage repeated practice from different perspectives, and to reinforce social interaction learning.

GenRole: Personalizing Role Play for Educators Supporting Autistic Students’ Social Interaction Learning

CHI ’26, April 13–17, 2026, Barcelona, Spain

Figure 7: (a) Structure of personalized components in GenRole across different modes. (b) Visual representation of components in Echo Mode, Dialogue Mode, Character Mode, and Exploration Mode

4.3.3 Personalize Visual Components. The AI-generated visual feature in GenRole received positive feedback, with its intuitive integration of different visual components during the role play. Participants particularly liked personalizing visuals based on children’s preferences (N=7), such as adding features like a favorite red hair clip, which made the role play more engaging and memorable (P4). They also thought that GenRole’s personalized visual styles can suit children’s preferences and enhance their experience (P13). As P4 noted, “The art style is adorable, and it was easy to generate characters with just a short prompt.” While the cartoon style appealed to autistic children, some participants felt the visuals might not be suitable for certain children with cognitive impairments, which would potentially reduce engagement (P14). A suggested improvement was to provide realistic images, such as real-world photography of familiar objects (N=5), to enhance their understanding (P3) and the transferability of learned skills to real-life scenarios (P11). Based on feedback from the pilot study, we added an “upload” option for the personalized visuals component. Teachers can choose between generating images or uploading custom visuals, based on the child’s cognitive level and preferences. This modification aims to help children with different learning needs better understand and engage with the scenarios.

4.3.4 Improving Language Expression. The use of simple, direct, and easy-to-understand language in role play scripts was essential for effective communication and children’s engagement. For example, instead of formal phrases like “You are making progress”, participants suggested using more natural expressions such as “You did great!” (P12) to make the dialogue more relatable and engaging for children (P3). Additionally, using more direct and instructional language made the message clearer for autistic children to understand. P16 shared her experience of revising a generated role play script—“Teacher, may I look into your eyes?”—which she found too formal and indirect for daily conversation. She modified it to “When you talk to the teacher, be polite and look at her”, making it more straightforward to understand. Some participants (N=5) recommended avoiding technical or abstract terms. P7 noted that eliminating subjective words like “might” or “perhaps” improved the clarity of the dialogue, aligning it better with the direct communication style typically used in daily life. Simplifying word choices

helped autistic children better understand and apply social interaction skills during role play. Additionally, due to varying levels of language comprehension, participants (N=9) suggested adding voiceovers and background sounds to enhance engagement and increase interactivity. Based on feedback from the pilot study, we revised our prompts by limiting sentence length and adjusting the expression style to make it simpler and easier to understand. The final version of our prompt is shown in the supplementary material. We also added voice functionality and background sounds to the role play stage, helping children with varying levels of language comprehension engage more fully and interactively.

5

Methods

We conducted a 2-week user study to empirically assess the impact of GenRole’s role play and personalized features on social interaction learning for autistic children and teachers. Hong Kong University of Science and Technology (Guangzhou) IRB approved this research study. All participants received compensation for their time in this study. The main user study (Fig. 2) aimed to answer the following questions: (1) What are the learning outcomes for children using GenRole? (2) How do GenRole’s personalized components and features affect role play based social interaction learning? (3) What challenges and opportunities arise in GenRole’s role play activities for autistic children?

5.1

Participants

A total of 12 teacher-student pairs were recruited for the experiment, one student withdrew because of health. All teachers were familiar with the students they had taught and had therapy experiences (see Table. 3 and Table. 4). We compensated each participant with gifts equivalent to 100 CNY (≈17 USD) per hour for attending the study. We recruited teachers and children from the [Anonymous] special education school using convenience sampling, based on the following criteria: (1) All children have been diagnosed with Autism Spectrum Disorder by authoritative institutions and hospitals. (2) All children need to have basic learning abilities, understanding around 100-200 Chinese characters. (3)All participating children were verbal and capable of producing spoken language, though

CHI ’26, April 13–17, 2026, Barcelona, Spain

Li et al.

Figure 8: The flow chart for each participant in the user study.

most required teacher prompts to initiate or sustain conversation. No participants were non-speaking or used augmentative and alternative communication (AAC) systems. (4) Not receiving courses on social interaction learning during the experiment period.

5.2

Study Procedure

Each participant in our user study goes through three parts, as shown in Fig. 8: (1) Introduction and Tutorial to Educators (1 hour). We introduced the study’s purpose and provided written guidelines outlining the experimental procedure. We also conducted a live demonstration to illustrate how to use GenRole, after which participants explored GenRole individually. (2) Pre-test (10 minutes) We asked teachers to rate the children’s social skills on a 5-point Likert scale based on their observations. Each teacher completed three questions (see Section 5.3.1) to assess their selected social skill and examine the students’ ability to express and apply the skill appropriately. (3) Experiment (3 classes within 1 week, each consisting of approximately 10 minutes for preparation and 20 minutes for role play class). Each teacher selects one social skill (Table. 4) to focus on throughout the experiment, and generates three new role play background scenes and role play scripts for each class, the experiment set up as shown in Fig. 9. The duration and sequence of the four usage modes are flexible. A researcher is onsite in each class to observe and record teacher–student interactions. (4) Post-test and interview (1 hour). After the three classes, teachers completed post-test questions and the 5-point Likert scale of the System Usability Scale (SUS) [70]. The post-test procedure was the same as the pre-test. After that, each teacher participated in a semi-structured interview, lasting approximately 40 minutes.

5.3

Data Collection and Analysis

In the experiment, we collected three types of data as follows:

5.3.1 Teacher’s Self-reported Feedback of Students’ Learning Outcome. We designed preand post-test rating questions, with the content tailored to the specific social skills selected by each teacher. Because the GenRole approach focuses on context-specific behaviors that are not well captured by broad instruments [15], we did not adopt existing standardized social-skills scales. Additionally, prior research suggests that global rating scales are often insensitive to changes in short-term interventions [105], making them not wellsuited to the brief duration of our study. Instead, we drew on prior research showing that LLM-generated scales can exhibit reliable psychometric properties [42, 105], particularly for cognitive and knowledge-based items [55]. Therefore, rather than adopting existing standardized social-skills scales, we used prompt-engineering techniques to generate an initial pool of questions. Three experienced special education teachers then reviewed our AI-generated

sample items to confirm their clarity, relevance, and content validity. Based on the teachers’ feedback, our scale-generation procedure was further informed by cognitive models of skill acquisition [104]. This process ensured that the focus was on supporting autistic children in understanding and engaging with the activities and reflecting on their emotional responses, rather than solely on observable behavioral performance. We also used the expert revised items as few-shot examples to structure prompts and generate scales for each selected skill. In their final review, the experts praised the iteratively refined scales and noted that this procedure aligns with standard school assessment practices. Example questions are provided in the supplementary material. We performed normality tests on the data sets. If the data were normally distributed, we used the t-test. If not (p < 0.05), we applied the paired-sample Wilcoxon test to assess the significance of the differences between the pre- & post-test. Additionally, we collected the SUS to verify GenRole’s usability, ensuring it is effective and user-friendly for teachers during the usage process.

5.3.2 Semi-structured Interviews. During the semi-structured interviews, we explored participants’ perspectives, expectations, and concerns regarding the use of GenRole. The interview outlines are provided in the supplementary material. Two researchers conducted a thematic analysis [18] of the interview transcripts. They began by generating initial codes to identify emerging themes, followed by iterative discussions to refine and ensure consistency. Any disagreements were resolved through consultation with a third expert (the last author) until consensus was reached. The thematic analysis was then applied to extract and categorize key insights. Results are presented in Sections 6.1 to 6.4.

6

Findings

We report our findings from the user study in three key aspects: (1) Qualitative and quantitative data on children’s learning outcomes through GenRole. (2) GenRole’s roleplay feature facilitates social interaction and learning. (3) Opportunities and challenges of using GenRole for personalized teaching.

6.1

Children’s Engagement and In-Study Performance

To summarize children’s engagement and in-study performance, we report that all participants completed the experiment. A total of 33 classes were conducted, totaling 627.57 minutes (M = 19.02, min = 10, max = 32.45, SD = 5.07), indicating sustained engagement with the GenRole social interaction learning content. In the pre & post test, reported as descriptive evidence of task performance, results from the paired Wilcoxon test (Fig. 9) showed post-test scores (M = 3.03, SD = 0.81) were significantly higher (z=4.35, p<0.01) than pretest scores (M = 2.18, SD = 0.73). These results suggest that children performed better on the selected tasks in the post-test. Regarding

GenRole: Personalizing Role Play for Educators Supporting Autistic Students’ Social Interaction Learning

CHI ’26, April 13–17, 2026, Barcelona, Spain

Table 3: Demographic information of teacher participants in the main user study.

Group

Participant

Gender

Age

Major

Teacher

P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11

F M F F F F F M M F F

28 27 27 26 28 29 25 27 31 27 30

Applied Psychology Rehab Therapy Rehab Therapy Special Education Special Education Special Education Special Education Applied Psychology Special Education Special Education Special Education

Experience

3 Years 3 Years 5 Years 5 Years 5 Years 6 Years 3 Years 3 Years 7 Years 5 Years 8 Years

Teaching

Grade 4: Life and Labor; Living Language Grades 1–3: Social Play Grades 1–3: Rehab Training Course Grade 4: Life and Labor; Living Language Grades 3–6: Living Language Grade 7: Mathematics Grades 5–6: Living Language Grade 5: Life Adaptation Course Grade 8: Life and Labor Grade 7: Life and Labor; Living Language Grade 5: Living Language

Table 4: Demographic information of student participants in the main user study.

Group

Children

Participant P1’s Student P2’s Student P3’s Student P4’s Student P5’s Student P6’s Student P7’s Student P8’s Student P9’s Student P10’s Student P11’s Student

Gender M M M M M M M M M M M

Age 10 10 10 9 15 13 12 11 14 14 11

Learning Experience 7 Years 7 Years 5 Years 6 Years 12 Years 11 Years 5 Years 9 Years 8 Years 11 Years 4 Years

usability, the SUS yielded an average score of 76.88, indicating that educators found GenRole to be highly usable, intuitive, and user-friendly.

Teacher

(a)

Student

(b)

Figure 9: (a) Pre & Post test scores. (b) Experiment set up.

6.2 Opportunity and Challenge for Genrole’s Role Play Feature in Enhancing Social Interaction Learning

6.2.1 Teachers’ Perspectives on GenRole. In our semi-structured interview, most teachers reported improvements in various aspects of social interactions through GenRole (N=12). P10 highlighted children’s improvements in their verbal communication skills, and P4 noted an increased willingness to express themselves. Some

Social Interaction Skill Sharing of Past Events Initiative to Share with Others Express Needs and Requirements Express Needs and Requirements Understanding What to do First and What To Do Later Sharing of Past Events Choose your favorite out of 3 Express Needs and Requirements Able to listen to others’ questions Initiative to Share with Others Choose your favorite out of 3

teachers observed that children were able to understand and learn the dialogue content (P11, P9, P7). As P11 stated, “In class, following the questions, he gradually understood that he needed to follow the dialogue and share his thoughts.” With the increase in training classes, most teachers also observed that GenRole helped deepen children’s understanding and reinforced the skills they were learning (N=7). For example, in the third class, P3 noted a reduction in off-topic answers and observed that children were more able to express their needs directly. These findings suggest that GenRole gradually enhances children’s social interaction skills. For some teachers, GenRole was perceived as more efficient than classic role-play lesson preparation (N=5), which often involves time-consuming content collection. P11 explained that teachers previously had to search through multiple storybooks and align them with teaching objectives, whereas with GenRole they could quickly generate short social stories to complete the lesson content. Similarly, P9 stated, “..This tool can reduce quite a bit of our workload. Normally, we collect many pictures and write a lot of text ourselves. With this system, we simply input our requirements, and it generates content automatically, making lesson preparation much easier.” P3 further noted that the tangible cards generated by GenRole can be printed out, streamlining preparation and helping children maintain attention. However, other teachers did not primarily view GenRole as an efficiency tool. P8 described it as an auxiliary tool for extending and deepening lessons, noting her extensive teaching experience had already provided a rich repertoire of materials.

CHI ’26, April 13–17, 2026, Barcelona, Spain

Li et al.

P6 shared her desired usage scenario: “In class, after students have finished learning a text, I would like to use that text to decide which roles I want them to play, and then generate (by GenRole) so that they can extend the lesson in a more immersive way.”

be reluctant to engage.” Therefore, in more abstract modes like Exploration Mode, sustaining focus and engagement requires higherlevel skills, and additional scaffolding can help children gradually develop these abilities.

6.2.2 GenRole’s Dialogue Structured Script Facilitates Role Play and Social Interaction. As mentioned in section 4.1.2, GenRole’s role play scripts were based on a dialogue structure personalized to selected social skills, ensuring that each conversation interaction follows a clear, organized sequence. Some teachers noted that this structured dialogue makes the communication process easier to understand, and helps children engage in role play more confidently (N=3). As P8 highlighted,

The Progressive Role Play Mode Supports Rehearsal and Enhances Generalization. GenRole’s four-mode design enables children to repeatedly rehearse selected social skills, helping them become more confident and proficient in applying these skills. This process supports the generalization of skills to real-life situations, as some teachers (N=4) observed that children began to apply the social interaction skills learned in the experiment to other daily life situations. P10 noted,

“...It’s about letting the children know we’re practicing a dialogue (in role play)...they understand when it’s their turn to speak and when it’s the teacher’s turn, making the structure clearer.”

“He (P10’s student) was reluctant to share, and the situation of distributing candy at the school gate happened in real life. He was unwilling to share and showed emotional behaviors. So when we returned to the experiment here, we did some training...and there was some communication and dialogue, such as ‘Would you like to try some?’ ‘Next time, I’ll share with you too.’ This communicative dialogue helped guide him, provided a buffer and rehearsal...rather than directly causing emotional problems.”

Moreover, GenRole’s dialogue structure fostered two-way social interactions between children and teachers, offering more interactive dialogue and instructive back-and-forth communication, compared to traditional storybooks (P4). P10 mentioned, “The scenarios and dialogues were quite instructive. During learning...children could participate in conversations, but they often didn’t know how to use positive language to express their needs. The dialogues provided here were helpful.” This highlighted that the use of an instructive and targeted dialogue structure in GenRole facilitated social interaction skills by encouraging children to express their needs with appropriate language.

6.2.3 Progressive Role Play Modes Support Rehearsal and Generalization.

A Progressive Role Play Method from Concrete to Abstract. GenRole’s four-mode design (Echo Mode, Dialogue Mode, Character Mode, Exploration Mode) was perceived by teachers as following a progressive learning structure. This progression moved from concrete tasks, such as simple repetition, to more abstract tasks, like open-ended questions, which aligned with children’s learning needs. Teachers found this progression of modes both logical and effective (N = 7), as it allowed them to gradually align teaching goals with children’s varying learning needs (P11). Similarly, P6 described her experiences:

“...it’s a process from something concrete to something abstract... at first, it’s response-based or choice-based. In the middle, it’s like moving from short sentences to longer ones, linking different responses together... and finally, you need to express what you feel in this scenario.”

In this progression, Echo Mode and Dialogue Mode were considered simpler, with more scene-based cues helping children stay focused (P1). Children performed well in both repeating and conversing because they could see the scripts and characters, which helped maintain their attention (P6, P7). However, for the more abstract scenarios (Exploration Mode), with open-ended questions, children found it harder to understand (P4, P7), which made it more difficult for them to remain engaged (P6). P4 stated,“The fourth scene is a bit difficult (open-ended questions), which causes children to

One suggestion was to expand the scenarios for practice. Teachers proposed adding a fifth mode to GenRole that incorporates different scenarios within the same storyline (N=2), thus broadening the range of situations in which children can practice and apply their skills. Another suggestion was to provide opportunities for children to practice at home (P1). Some teachers (N=4) emphasized the importance of collaborating with parents to assign home-based exercises, which could reinforce learning and support real-life practice.

6.2.4 Role Play Interactive Stimuli-Triggers Attention and Motivation in Autistic Children. GenRole’s digital system engages autistic children by using rewards and reinforcers as key interactive stimuli to trigger attention and engage their interest (N=4). For example, in the Reward Page, the reinforcers and encouraging phrases as triggers can help sustain children’s focus (P11). Additionally, the background scene settings in GenRole can be easily interchangeable, which is considered positive stimuli, to broaden the scope of learning and help children better understand the story (P10, P11). However, teachers express concern that the background scene should be more dynamic and continuously evolving to better trigger the attention of autistic children (N=3). For example, dynamic scene transitions (P5) and video playback (P4) could be incorporated. Additionally, adding interactive elements that children could engage with (e.g., soccer balls in a playground scene that children could click on) could have helped enhance their focus (P6, P5). Tangible cards can also serve as a trigger, providing a multisensory learning experience and offering an interactive method for teaching social interaction skills in GenRole (P4). The majority of teachers (N=9) believe that tangible cards facilitate engagement and present key interactive items (P10). Additionally, tangible props assist students in learning social skills (P2) by acting as an auxiliary tool, offering a physical element in an otherwise virtual environment (P8).

GenRole: Personalizing Role Play for Educators Supporting Autistic Students’ Social Interaction Learning

P7 stated: “... this method (tangible card) is better for autistic children than the screen because they can have it right in front of them. For instance, they may not focus on something far away, but if it is brought closer, they will recognize it as an object they need to observe, and it draws their attention.” Similarly, P3 expressed: “I think the tangible card can reduce their random clicking on the computer, making it easier to control and focus their attention on the card.” While tangible cards benefit many children, their effectiveness depends on individual preferences, highlighting the need for personalized use. Two teachers noted that tangible cards may not be suitable for all children. P4 mentioned her student enjoys tapping objects, so he doesn’t respond well to tangible cards. Similarly, for some children, tangible cards may trigger problem behaviors, such as tearing paper (P8).

6.3

Opportunities and Challenges of Personalizing Components in Role Play Activities

6.3.1 Flexible Role Play Scripts Tailored to Children’s Unique Needs. Role play scripts were a key feature of GenRole, offering personalized stories with meaningful instructional value (P6), making them suitable for students with diverse learning needs (P1, P3, P10). As P11 mentioned, “...It (Role play scripts) can also be modified based on students’ specific needs in different areas, making it suitable for a wide range of children.” Despite this benefit, P2 pointed out limitations in the alignment of GenRole’s scripts with children’s cognitive level, The lack of a temporal framework is a key aspect. Teachers mentioned the need for personalized story generation from the perspectives of ‘past, present, and future,’ based on the cognitive stages of children. As P8 stated,

“...It’s already difficult for him to express the present, so retelling past experiences is even harder. Recounting past emotions or events is a more advanced task...If describing what he wants to do in the future, it becomes even more difficult because he can’t distill those experiences.”

This highlights the need for more granular personalization (P8), allowing for tailored content generation with personalized options to better align with the child’s developmental stage and unique needs. Another benefit of GenRole is its ability to generate content tailored to the teacher’s specifications and closely linked to the children’s life experiences (P4). Teachers found that including familiar or real-life events in role play scripts enhanced children’s engagement (N=4), fostered a sense of immersion (P5), and increased their interest (P7). However, this level of personalization also depends heavily on the teacher’s understanding of each child’s unique characteristics (P3, P4). As P8 stated, “...This is where AI faces a challenge—teachers must know the children’s traits to create effective lesson content”, highlighting the importance of teachers’ understanding of students’ needs and experiences.

6.3.2 Generative Visual Components Improve Efficiency in Personalized Activity. Teachers praised the visual components, especially the background scenes, for being easy to generate and personalize, which made the preparation process more efficient compared to

CHI ’26, April 13–17, 2026, Barcelona, Spain

searching for materials such as videos, images, and picture books (N=4). As P6 mentioned, “Being able to generate images directly is very helpful. I don’t have to search for them myself, and even when I do, I can’t always find something I like. With direct generation, it’s much better—I can adjust them as needed.” GenRole was also regarded as maintaining a consistent cartoon visual style during personalization (N=7), which helped children understand the story better (N=6). P3 shared her experience of facing style issues in traditional teaching while using a widely-used generative AI tool, ERNIE Bot 4 , “I was working on a class about a bear inviting guests recently, and I needed an image of a bear eating a pork bone. Just finding the pork bone alone took me a long time. The results were either too raw, too bloody, or the hands didn’t look right. I used ERNIE Bot to generate it, but it still didn’t turn out well.” Although GenRole adopted a unified cartoon visual style, some teachers (N=3) found it unsuitable for children with cognitive impairment, as the images did not align with their real-life experiences. Instead, this style appeared more suitable for children without cognitive impairment (P5). Many teachers preferred adding more realistic background scenes (N=7) or incorporating real-life images (N=5) as personalization options. As P12 mentioned, “For students, given the varying levels of cognitive content, we can categorize visuals into different gradients. Some visuals should be concrete objects and realistic pictures, followed by photographs, then cartoon images or simplified sketches, and finally, text.” These findings suggest that, as autistic children have varying learning needs, more personalized visual settings are necessary. Therefore, future designs should balance cartoon and realistic styles while considering the cognitive load of autistic children at different levels.

7 Discussion 7.1 Supporting Educators with Personalized Role Play for Autistic Students’ Social Interaction Learning

In the role-play stage, our study examines how traditional role play activities, commonly used by educators to support social interaction learning [13, 53], can be redesigned as a progressive learning system. GenRole integrates insights from previous work on role play design and extends them by improving usability and adaptability for educators. Findings show that these progressive methods were considered effective by educators in supporting autistic students, enabling them to begin with simple and concrete scenarios [41]. The increasing difficulty is aligned with their developmental pace and learning needs [101]. Additionally, structured dialogue [80] serving as clear cues in GenRole has been shown to help children engage in learning. Similar to previous research, clear cues and information in role play effectively direct participants on what to do, thereby facilitating better understanding [45]. Regarding personalization, previous research has shown that mapping physical-world actions onto role play characters helps reflect the user’s identity in the character [45]. In GenRole, we integrate generative AI to help educators map autistic students’

4 https://yiyan.baidu.com/

CHI ’26, April 13–17, 2026, Barcelona, Spain

real-life experiences into role play content, exploring how such personalization can support engaging students more effectively. Unlike EmoEden [101], which offers fixed scenarios (e.g., home, park) and predefined characters for dialogue, GenRole allows teachers to generate personalized characters and scenes tailored to the students’ diverse needs and life experiences. This continuous practice across multiple scenes also has the potential to facilitate behavior rehearsal and generalization. While previous tools such as EMoooly [78] focused on cartoon-style characters, GenRole allows teachers to describe character features or upload real-life photos, enabling the personalization of characters, background scenes, and reinforcers, which are key components of typical role play processes [100]. Our findings about GenRole’s personalization feature align with prior research indicating that AI-based tools can support real-world behavioral improvement in autistic children [89]. Building on this, GenRole demonstrates the potential to generate personalized content for social interaction learning, moving beyond generic story templates. By integrating role-play stories with personalized visuals, GenRole offers a learning experience better suited to each child’s unique developmental needs. This capability directly supports educators in creating individualized learning materials that align with each student’s unique interests, behavioral goals, and cognitive levels [25, 65], thereby addressing persistent challenges such as time constraints, limited resources [73, 108], high workload pressures, and competing academic demands [1].

7.2

Design Insights from Educators’ Perspective on Role Play for Autistic Students’ Social Interaction

7.2.1 Personalization Tailored to Cognitive Needs from Abstract to Concrete. Personalizing visual elements to reflect children’s preferences and experiences can enhance both engagement and motivation. During our user study, teachers noted that GenRole’s generative component should allow for selecting between complex and simplified elements to better accommodate individual differences, thereby enabling a more tailored learning experience. Prior research also emphasized the importance of adapting story difficulty based on children’s proficiency levels [78, 101]. Therefore, future development should focus on implementing personalization strategies that range from abstract (complex) to concrete (simplified) representations, supporting the diverse cognitive needs of autistic children.

7.2.2 Stimulating Interaction as a Trigger for Attention. Previous research has highlighted the importance of attending to visual and interactional details in role play for social interaction learning [45]. As autistic children tend to be visual learners [90] and may disengage when tasks become repetitive [35], incorporating visual and interactive stimuli in role play could help sustain their attention. While GenRole offers some degree of stimulation (e.g., sound and cartoon-style visuals), teachers suggested that additional interactive components could further help maintain focus (N=5). In particular, incorporating interaction and animation into role play activities may enhance external stimulation and boost engagement (P3). Future design could explore allowing children to manipulate AI-generated images, such as dragging elements to build the story

Li et al.

structure themselves [49] or integrating timed interactions (e.g., double tap, long press, multi-touch gestures) [51, 89]. Other possibilities include incorporating AR [78] or haptic feedback [40, 50] to further support attention and engagement.

7.2.3 Behavior Rehearsal Promotes Generalization through Roleplay. GenRole allows for selecting different social interaction skills, enabling the generation of diverse role play contexts. In our user study, most participants commended GenRole’s impact on improving students’ behavior through its four role play modes. Teachers likened these modes to behavior rehearsal, where children practice desired social interaction skills under teacher guidance to refine social responses into more effective ones [67]. In GenRole, this concept is integrated with AI-driven feedback, allowing teachers to generate and simulate specific social contexts, rehearse various social responses repeatedly, and adjust based on the children’s feedback. This iterative practice process promotes the generalization of learned social interaction skills to real-life situations, reinforcing children’s adaptive responses [19].

7.3

Considering Cultural and Educational Contexts in Autism

The cultural context, through different beliefs and social norms, shapes the social acceptance of autistic people [3, 28] and influences autistic children’s access to education and support [5, 64, 93]. While we adapted GenRole’s social scenarios to Chinese social norms and communication patterns, educational provision for autistic children varies widely across regions, so the effects observed in our study may not generalise globally. When tools are introduced into new countries, knowledge, values, and expectations are often assumed rather than co-constructed with local stakeholders, limiting cultural relevance [5]. Given that many AI technologies are developed in Western, high-income settings and primarily benefit those contexts [11], generative AI tools for autism need to be adapted to local cultural norms, values, and communication styles [2, 71], as well as structural factors such as affordability and access to digital infrastructure [2]. Additionally, the appearance of autistic traits varies across cultures. Research shows that preoccupation with parts of objects is more common among white American children [95]. In our study, we observed that educators often used reinforcers as key interactive stimuli to trigger attention and engage children’s interest. Whereas in Nigeria, such behaviors are seen in only a small minority of autistic children [12]. These contrasts illustrate that cultural context shapes practice, so tools like GenRole must be interpreted and adapted accordingly. Educational provision for autistic children varies substantially across regions. For example, in Ghana, autistic children are often systematically excluded from mainstream inclusive schools, and teachers have limited autonomy to adopt personalised learning [5]. Individual education plans (IEPs) further illustrate how opportunities for personalisation differ across cultural and national contexts. In Ghana, IEPs are often developed with little regard for local school realities and with minimal involvement from parents and students, so they seldom provide personalisation [5]. In Lithuania, many students still do not have an IEP at all, further limiting access to individualised support [61]. These patterns point to broader challenges in some regions, including limited understanding of autism, low

GenRole: Personalizing Role Play for Educators Supporting Autistic Students’ Social Interaction Learning

participation of autistic children in educational decision-making, inadequate teacher training, and shortages of specialists [5]. Such conditions may constrain the mutual understanding required to use tools like GenRole effectively in these contexts. Similarly, the educational environment shapes autistic children’s experiences. In mainstream schools, which often adopt a “onesize-fits-all” approach, autistic children may suppress their natural responses, negatively affecting their learning [24]. By contrast, special education schools typically offer more individualised support: teachers are more receptive to personalised methods, and staff respond to each case with tailored strategies, facilitating achievement and reducing anxiety triggers [24]. In our study, conducted in a special education setting, this personalised orientation likely made it easier for educators to adopt GenRole and to accommodate children’s individual needs, which in turn helped children adapt to personalised methods. In summary, cultural differences and educational contexts shape both the manifestation of autistic traits and the implementation of interventions, underscoring the need to tailor GenRole to different settings in future research.

7.4

Limitations and Future Work

This research has several limitations that can be addressed in future studies. First, due to the winter holiday in China, the user study involved only three classes per child, which may not be sufficient to observe long-term effects or sustained improvements. The preand post-test items were LLM-assisted, generated via a standardized prompt and teacher-reviewed for relevance and clarity. Future work should strengthen the evaluation framework by validating LLM-generated outcomes against established scales. Researchers should clearly report the item-generation pipeline to support alignment and reproducibility across studies, given concerns such as potential plagiarism of existing scales in LLM-generated items. Additionally, all participants in the user study were boys, which limits the generalizability of the findings to broader student populations. Furthermore, since all participants were recruited from schools in China, the cultural and educational context may also influence the generalizability of the results. Future research should investigate the effectiveness of GenRole in diverse cultural and educational settings. Moreover, while the personalization process in GenRole positively improved children’s social interaction and learning, future research should incorporate mobetter to support the varied learning needs of autistic children, and include systematic classroom video observations to better support the varied learning needs of autistic children. To further support generalization, future designs could incorporate real-world scenarios or projection-based immersive environments [21], helping children apply what they have learned in a more engaging and realistic context.

8

Conclusion

In this study, we present GenRole, a system that incorporates progressive role play and enables teachers to create personalized components tailored to the diverse needs of autistic children, to enhance their social interaction skills. GenRole highlights the integration of generative AI and role play in social interaction training, empowering teachers to prompt effective learning outcomes for children. Our findings demonstrate that GenRole’s role play feature effectively

CHI ’26, April 13–17, 2026, Barcelona, Spain

enhances social interaction skill acquisition, and the personalized components in role play training can adapt to children’s learning processes. We also provide valuable design insights and implications for developing future generative AI tools that better support autistic children’s learning engagement. Future work will explore expanding personalization parameters and incorporating interactive triggers to further enhance the adaptability of the design.

9

Disclosure about Use of Large Language Models

We employed large language models in the following ways: (1) Using the GPT API (ChatGPT) to generate sample content, including role play scripts and pre-& post-test questions. After generation, we carefully reviewed these outputs to ensure their appropriateness. (2) Using ChatGPT to support programming during implementation. (3) Using ChatGPT for translating the interview guide and scale items we developed. The translation results are attached in the supplementary material after inspection. (4) Using DALL·E 3 to generate system visuals and black-and-white line-drawing character illustrations for the teaser figure. Details of these uses are provided in the relevant sections. The authors take full responsibility for the outputs and the use of AI in this paper.

Acknowledgments

This paper is supported by China NSFC RFIS Grant (No.W2533160); Guangdong Provincial Key Lab of Integrated Communication, Sensing and Computation for Ubiquitous Internet of Things (No.2023B12 12010007); 111 Center (No.D25008); Guangzhou Municipal Bureau of Education and the Guangzhou Education Foundation (No.202512 13789); Department of Education of Guangdong Province (No.SZFN TSJY2025002); Guangzhou Science and Technology Plan Project (No.2025A03J0172); The ‘Five Fifths’ Project (No.2023WW704). We are grateful to all the participants for their invaluable time and contributions to this research. We also thank Yuchong Liu for sharing his autism-related domain knowledge.

References

[1] Harriet Able, Melissa A Sreckovic, Tia R Schultz, Justin D Garwood, and Jessica Sherman. 2015. Views from the trenches: Teacher and student supports needed for full inclusion of students with ASD. Teacher Education and Special Education 38, 1 (2015), 44–57. [2] Oyeyemi Patricia Adako, Oluwafemi Clement Adeusi, and Peter Adeniyi Alaba. 2024. Revolutionizing autism education: harnessing AI for tailored skill development in social, emotional, and independent learning domains. Journal of Computational and Cognitive Engineering 3, 4 (2024), 348–359. [3] Omniah AlQahtani and Maria Efstratopoulou. 2025. The UAE and Gulf countries’ cultural characteristics and their influence on autism. Review Journal of Autism and Developmental Disorders 12, 1 (2025), 163–167. [4] Henry Angulo, Michelle Chan, and Laura DeThorne. 2019. Life is a stage: Autistic perspectives on neurotypicality. Autism in Adulthood 1, 4 (2019), 276–285. [5] Jane H Anthony. 2010. Towards inclusion: influences of culture and internationalisation on personhood, educational access, policy and provision for students with autism in Ghana. Ph. D. Dissertation. University of Sussex. [6] Victor Nikhil Antony and Chien-Ming Huang. 2024. ID. 8: Co-Creating visual stories with Generative AI. ACM Transactions on Interactive Intelligent Systems 14, 3 (2024), 1–29. [7] Prabal Datta Barua, Jahmunah Vicnesh, Raj Gururajan, Shu Lih Oh, Elizabeth Palmer, Muhammad Mokhzaini Azizan, Nahrizul Adib Kadri, and U Rajendra Acharya. 2022. Artificial intelligence enabled personalised assistive tools to enhance education of children with neurodevelopmental disorders—a review. International Journal of Environmental Research and Public Health 19, 3 (2022), 1192.

CHI ’26, April 13–17, 2026, Barcelona, Spain

[8] Nirit Bauminger and Connie Kasari. 2000. Loneliness and friendship in highfunctioning children with autism. Child development 71, 2 (2000), 447–456. [9] Nirit Bauminger and Connie Kasari. 2000. Loneliness and friendship in highfunctioning children with autism. Child development 71, 2 (2000), 447–456. [10] Rongqi Bei, Yajie Liu, Yihe Wang, Yuxuan Huang, Ming Li, Yuhang Zhao, and Xin Tong. 2024. StarRescue: the Design and Evaluation of A Turn-Taking Collaborative Game for Facilitating Autistic Children’s Social Skills. In Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems (Honolulu, HI, USA) (CHI ’24). Association for Computing Machinery, New York, NY, USA, Article 67, 19 pages. https://doi.org/10.1145/3613904.3642829 [11] Mohammad A Beirat, Ahmad Algolaylat, Hussein Al Njadat, Bassam AlAbdallat, and Alaa K Al-Makhzoomy. 2025. Utilization of Artificial Intelligence and Assistive Technology in Autism: Diagnosis, Treatment, and Education Applications–A Systematic Literature Review. Educational Process: International Journal 17 (2025), e2025350. [12] MA Bello-Mojeed, OO Omigbodun, MO Bakare, and AO Adewuya. 2017. Pattern of impairments and late diagnosis of autism spectrum disorder among a subSaharan African clinical population of children in Nigeria. Global Mental Health 4 (2017), e5. [13] M Bevčič, J Rugelj, and S Jedrinović. 2024. INNOVATIVE TEACHING THROUGH ROLE-PLAYING AND STORYTELLING WITH SERIOUS GAMES. In EDULEARN24 Proceedings. IATED, 3953–3960. [14] Laura Boccanfuso, Erin Barney, Claire Foster, Yeojin Amy Ahn, Katarzyna Chawarska, Brian Scassellati, and Frederick Shic. 2016. Emotional robot to examine different play patterns and affective responses of children with and without ASD. In 2016 11th ACM/IEEE International Conference on Human-Robot Interaction (HRI). IEEE, 19–26. [15] Sven Bölte, Fritz Poustka, and John N Constantino. 2008. Assessing autistic traits: cross-cultural validation of the social responsiveness scale (SRS). Autism Research 1, 6 (2008), 354–363. [16] Monique Botha, Jacqueline Hanlon, and Gemma Louise Williams. 2023. Does language matter? Identity-first versus person-first language use in autism research: A response to Vivanti. Journal of autism and developmental disorders 53, 2 (2023), 870–878. [17] Fatima A Boujarwah, Hwajung Hong, Rosa I Arriaga, Gregory D Abowd, and Jackie Isbell. 2010. Training social problem solving skills in adolescents with high-functioning autism. In 2010 4th International Conference on Pervasive Computing Technologies for Healthcare. IEEE, 1–9. [18] Virginia Braun and Victoria Clarke. 2012. Thematic analysis. American Psychological Association. [19] CYNTHIA LYNNE BROWNSMITH. 1976. The skill acquisition model: behavior rehearsal as a method for developing pro-social adaptive behaviors in elementary school children. Indiana University. [20] Ruth Busby, Rebecca Ingram, Rhonda Bowron, Jan Oliver, and Barbara Lyons. 2012. Teaching elementary children with autism: Addressing teacher challenges and preparation needs. The Rural Educator 33, 2 (2012), 27–35. [21] Yancheng Cao, Yangyang He, Yonglin Chen, Menghan Chen, Shanhe You, Yulin Qiu, Min Liu, Chuan Luo, Chen Zheng, Xin Tong, et al. 2025. Designing LLMsimulated Immersive Spaces to Enhance Autistic Children’s Social Affordances Understanding in Traffic Settings. In Proceedings of the 30th International Conference on Intelligent User Interfaces. 519–537. [22] Timothy C Clapper. 2010. Role play and simulation. The Education Digest 75, 8 (2010), 39. [23] Melinda L Combs and Diana Arezzo Slaby. 1977. Social-skills training with children. In Advances in Clinical Child Psychology: Volume 1. Springer, 161–201. [24] Anna Cook and Jane Ogden. 2022. Challenges, strategies and self-efficacy of teachers supporting autistic pupils in contrasting school settings: a qualitative study. European journal of special needs education 37, 3 (2022), 371–385. [25] Amy Shannon Cook, Steven P. Dow, and Jessica Hammer. 2017. Towards Designing Technology for Classroom Role-Play. In Proceedings of the Annual Symposium on Computer-Human Interaction in Play (Amsterdam, The Netherlands) (CHI PLAY ’17). Association for Computing Machinery, New York, NY, USA, 241–251. https://doi.org/10.1145/3116595.3116632 [26] Meg Cramer, Sen H Hirano, Monica Tentori, Michael T Yeganyan, and Gillian R Hayes. 2011. Classroom-based assistive technology: collective use of interactive visual schedules by students with autism.. In CHI, Vol. 11. 1–10. [27] Ana Paula de Carvalho, Camila S Braz, Sibele M dos Santos, Renato AC Ferreira, and Raquel O Prates. 2024. Serious games for children with autism spectrum disorder: A systematic literature review. International Journal of Human–Computer Interaction 40, 14 (2024), 3655–3682. [28] Anne de Leeuw, Francesca Happé, and Rosa A Hoekstra. 2020. A conceptual framework for understanding the cultural and contextual factors on autism across the globe. Autism Research 13, 7 (2020), 1029–1050. [29] Lara Delmolino and Sandra L Harris. 2012. Matching children on the autism spectrum to classrooms: A guide for parents and professionals. Journal of Autism and Developmental Disorders 42 (2012), 1197–1204. [30] Sebastian Deterding and José P Zagal. 2018. The many faces of role-playing game studies. In Role-playing game studies. Routledge, 1–16.

Li et al.

[31] Briano Di Rezze, Peter Rosenbaum, Lonnie Zwaigenbaum, Mary Jo Cooley Hidecker, Paul Stratford, Martha Cousins, Chantal Camden, and Mary Law. 2016. Developing a classification system of social communication functioning of preschool children with autism spectrum disorder. Developmental Medicine & Child Neurology 58, 9 (2016), 942–948. [32] Riddhi Divanji, Aayushi Dangol, Ella J. Lombard, Katharine Chen, and Jennifer D. Rubin. 2024. TogetherTales RPG: Prosocial Skill Development Through Digitally Mediated Collaborative Role-Playing. In Proceedings of the 23rd Annual ACM Interaction Design and Children Conference (Delft, Netherlands) (IDC ’24). Association for Computing Machinery, New York, NY, USA, 1012–1015. https://doi.org/10.1145/3628516.3662048 [33] Ana D Dueñas, Sophia R D’Agostino, and Joshua B Plavnick. 2021. Teaching young children to make bids to play to peers with autism spectrum disorder. Focus on Autism and Other Developmental Disabilities 36, 4 (2021), 201–212. [34] Fifth Edition et al. 2013. Diagnostic and statistical manual of mental disorders. Am Psychiatric Assoc 21, 21 (2013), 591–643. [35] Lizbeth Escobedo, Monica Tentori, Eduardo Quintana, Jesus Favela, and Daniel Garcia-Rosas. 2014. Using augmented reality to help children with autism stay focused. IEEE Pervasive Computing 13, 1 (2014), 38–46. [36] Min Fan, Xinyue Cui, Jing Hao, Renxuan Ye, Wanqing Ma, Xin Tong, and Meng Li. 2024. StoryPrompt: Exploring the Design Space of an AI-Empowered Creative Storytelling System for Elementary Children. In Extended Abstracts of the CHI Conference on Human Factors in Computing Systems. 1–8. [37] Isser Troy Gagan, Maria Angela Mikaela Matias, Ivy Tan, Christianne Marie Vinco, and Ethel Ong. 2023. Preparing Children with Level 1 ASD for Social Interactions through Storytelling with Amy: An Exploratory Study. In Extended Abstracts of the 2023 CHI Conference on Human Factors in Computing Systems. 1–7. [38] Isser Troy Mangin Gagan, Maria Angela Mikaela Eusebio Matias, Ivy Tan, Christianne Marie Vinco, Ethel Ong, and Ron Resurreccion. 2022. Designing a virtual talking companion to support the social-emotional learning of children with ASD. In Proceedings of the 21st Annual ACM Interaction Design and Children Conference. 464–471. [39] Elisa Gagnon. 2002. Power cards: Using special interests to motivate children and youth with Asperger syndrome and autism. AAPC Publishing. [40] Narayan Ghiotti, David Clulow, Serene Cheon, Kevin Cui, and Hyo Kang. 2023. Prototyping Kodi: Defining Design Requirements to Develop a Virtual Chat-bot for Autistic Children and Their Caregivers. In Companion Publication of the 2023 Conference on Computer Supported Cooperative Work and Social Computing (Minneapolis, MN, USA) (CSCW ’23 Companion). Association for Computing Machinery, New York, NY, USA, 126–131. https://doi.org/10.1145/3584931. 3606958 [41] Tom Giraud, Brian Ravenet, Chi Tai Dang, Jacqueline Nadel, Elise Prigent, Gael Poli, Elisabeth Andre, and Jean-Claude Martin. 2021. “Can you help me move this over there?”: training children with ASD to joint action through tangible interaction and virtual agent. In Proceedings of the fifteenth international conference on tangible, embedded, and embodied interaction. 1–12. [42] Friedrich M Götz, Rakoen Maertens, Sahil Loomba, and Sander van der Linden. 2023. Let the algorithm speak: How to use neural networks for automatic item generation in psychological scale development. Psychological Methods (2023). [43] Carol A Gray and Joy D Garand. 1993. Social stories: Improving responses of students with autism with accurate social information. Focus on autistic behavior 8, 1 (1993), 1–10. [44] Frank Gresham and Stephen N Elliott. 2008. Social skills improvement system (SSIS) rating scales. Bloomington, MN: Pearson Assessments. [45] Saumya Gupta, Theresa Jean Tanenbaum, Meena Devii Muralikumar, and Aparajita S Marathe. 2020. Investigating roleplaying and identity transformation in a virtual reality narrative experience. In Proceedings of the 2020 CHI Conference on Human Factors in Computing Systems. 1–13. [46] Sharon A Gutman, Emily I Raphael, Leila M Ceder, Arshi Khan, Katherine M Timp, and Sabrina Salvant. 2010. The Effect of a motor-based, social skills intervention for adolescents with high-functioning autism: two single-subject design cases. Occupational therapy international 17, 4 (2010), 188–197. [47] Andrew G Guzick, Sophie C Schneider, Philip C Kendall, Jeffrey J Wood, Connor M Kerns, Brent J Small, Ye Eun Park, Sandra L Cepeda, and Eric A Storch. 2022. Change during cognitive and exposure phases of cognitive–behavioral therapy for autistic youth with anxiety disorders. Journal of consulting and clinical psychology 90, 9 (2022), 709. [48] Anne Håkansson and Gloria Phillips-Wren. 2024. Generative AI and Large Language Models-Benefits, Drawbacks, Future and Recommendations. Procedia Computer Science 246 (2024), 5458–5468. [49] Ariel Han and Zhenyao Cai. 2023. Design implications of generative AI systems for visual storytelling for young learners. In Proceedings of the 22nd Annual ACM Interaction Design and Children Conference. 470–474. [50] Liwen He, Zichun Guo, Yanru Mo, Yue Wen, and Yun Wang. 2025. Exploring embodied emotional communication: a human-oriented review of mediated social touch. CCF Transactions on Pervasive Computing and Interaction (2025), 1–25.

GenRole: Personalizing Role Play for Educators Supporting Autistic Students’ Social Interaction Learning

[51] Liwen He, Yizhen Wu, Yu Li, Jing Pu, Weicheng Zheng, Yuling Sun, Min Fan, Yuhang Zhao, and Xin Tong. 2024. PixelMap: Enhancing Children’s Geography Learning with Tangible Augmented Reality Interaction. In Proceedings of the Twelfth International Symposium of Chinese CHI. 384–404. [52] Kate A Helbig. 2019. Evaluation of a role-playing game to improve social skills for individuals with ASD. (2019). [53] Kate A Helbig, Stefanie R Schrieber, Keith C Radley, and James R Derieux. 2024. Effects of a teacher-implemented social skills intervention for elementary students with autism and developmental disabilities. Journal of Educational and Psychological Consultation 34, 3 (2024), 210–238. [54] Niklas Holzner, Sebastian Maier, and Stefan Feuerriegel. 2025. Generative AI and Creativity: A Systematic Literature Review and Meta-Analysis. arXiv preprint arXiv:2505.17241 (2025). [55] Björn E Hommel, Franz-Josef M Wollang, Veronika Kotova, Hannes Zacher, and Stefan C Schmukle. 2022. Transformer-based deep neural language modeling for construct-specific automatic item generation. psychometrika 87, 2 (2022), 749–772. [56] Manying Hsieh, Hui-Ting Wang, Yu-ping Chen, and Wei-cheng Wang. 2025. Reliability and Validity of the Chinese Version of the Social Skills Improvement System Autism Spectrum Scale. Journal of Autism and Developmental Disorders (2025), 1–12. [57] Ltd iFLYTEK Co. 2021. iFLYTEK Open Platform. https://global.xfyun.cn/. Accessed: 2025-04-07. [58] Mega Iswari, Elsa Efrina, Arisul Mahdi, et al. 2019. Developing Social Skills of Autistic Children through Role Play. In 1st Non Formal Education International Conference (NFEIC 2018). Atlantis Press, 64–68. [59] Yewon Jin, SeonYul Lee, SeoHyeong Kim, Jiyeon Seo, Kyuha Jung, Hajin Lim, and Joonhwan Lee. 2023. DiVRsity: Design and Development of Group RolePlay VR Platform for Disability Awareness Education. In Proceedings of the 2023 ACM Designing Interactive Systems Conference (Pittsburgh, PA, USA) (DIS ’23). Association for Computing Machinery, New York, NY, USA, 161–174. https: //doi.org/10.1145/3563657.3596047 [60] Sinuo Jing, Bozhen Zhu, and Zaiqiao Ye. 2025. Lend A Hand: Designing A Robot for Teaching Social Skills to Children with High-Functioning Autism. In 2025 20th ACM/IEEE International Conference on Human-Robot Interaction (HRI). IEEE, 1368–1372. [61] Irena Kaffemaniene and Zivile Kulese. 2021. Ways of individualization of education for children with autism spectrum disorders: the experience of special pedagogues’. In SOCIETY. INTEGRATION. EDUCATION. Proceedings of the International Scientific Conference, Vol. 3. 37–50. [62] Steven K Kapp. 2020. Autistic community and the neurodiversity movement: Stories from the frontline. Springer Nature. [63] Fengfeng Ke, Jewoong Moon, and Zlatko Sokolikj. 2022. Virtual reality–based social skills training for children with autism spectrum disorder. Journal of Special Education Technology 37, 1 (2022), 49–62. [64] Hyun Uk Kim. 2012. Autism across cultures: Rethinking autism. Disability & Society 27, 4 (2012), 535–545. [65] Emma Kinnaird, Catherine Stewart, and Kate Tchanturia. 2019. Investigating alexithymia in autism: A systematic review and meta-analysis. European Psychiatry 55 (2019), 80–89. [66] Lynn Kern Koegel, Kristen Ashbaugh, Robert L Koegel, Whitney J Detar, and April Regester. 2013. Increasing socialization in adults with Asperger’s syndrome. Psychology in the Schools 50, 9 (2013), 899–909. [67] Arnold A Lazarus. 2002. Behavior rehearsal. Editors-in-Chief (2002), 253. [68] I-Jui Lee. 2021. Kinect-for-windows with augmented reality in an interactive roleplay system for children with an autism spectrum disorder. Interactive Learning Environments 29, 4 (2021), 688–704. [69] I-Jui Lee, Ling-Yi Lin, Chien-Hsu Chen, and Chi-Hsuan Chung. 2018. How to create suitable augmented reality application to teach social skills for children with ASD. State of the art virtual reality and augmented reality knowhow 8 (2018), 119–138. [70] James R Lewis. 2018. The system usability scale: past, present, and future. International Journal of Human–Computer Interaction 34, 7 (2018), 577–590. [71] Guang Li, Mohammad Amin Zarei, Goudarz Alibakhshi, and Akram Labbafi. 2024. Teachers and educators’ experiences and perceptions of artificial-powered interventions for autism groups. BMC psychology 12, 1 (2024), 199. [72] Yixuan Li and Lei Xue. 2023. Children’s Toy Design Process Based on the KanoAHP-FBS Model: Case of Multisensory Educational Toys. In 2023 International Conference on Culture-Oriented Science and Technology (CoST). IEEE, 311–316. [73] Sally Lindsay, Meghann Proulx, Nicole Thomson, and Helen Scott. 2013. Educators’ challenges of including children with autism spectrum disorder in mainstream classrooms. International Journal of Disability, Development and Education 60, 4 (2013), 347–362. [74] Di Liu, Hanqing Zhou, and Pengcheng An. 2024. " When He Feels Cold, He Goes to the Seahorse"—Blending Generative AI into Multimaterial Storymaking for Family Expressive Arts Therapy. In Proceedings of the CHI Conference on Human Factors in Computing Systems. 1–21.

CHI ’26, April 13–17, 2026, Barcelona, Spain

[75] Di Liu, Hanqing Zhou, and Pengcheng An. 2024. "When He Feels Cold, He Goes to the Seahorse"—Blending Generative AI into Multimaterial Storymaking for Family Expressive Arts Therapy. In Proceedings of the CHI Conference on Human Factors in Computing Systems (Honolulu, HI, USA) (CHI ’24). Association for Computing Machinery, New York, NY, USA, Article 118, 21 pages. https: //doi.org/10.1145/3613904.3642852 [76] Brittany Londer. 2023. Evaluating the Efficacy of Video Self-Modeling Social Skills in Children With Autism Spectrum Disorder. Ph. D. Dissertation. Walden University. [77] Catherine Lord, Mayada Elsabbagh, Gillian Baird, and Jeremy VeenstraVanderweele. 2018. Autism spectrum disorder. The lancet 392, 10146 (2018), 508–520. [78] Yue Lyu, Di Liu, Pengcheng An, Xin Tong, HUAN Zhang, KEIKO Katsuragawa, and JIAN Zhao. 2024. EMooly: Supporting Autistic Children in Collaborative Social-Emotional Learning with Caregiver Participation through Interactive AI-infused and AR Activities. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies 8, 4 (2024), 1–36. [79] Mark Matthews, Geri Gay, and Gavin Doherty. 2014. Taking part: role-play in the design of therapeutic systems. In Proceedings of the SIGCHI conference on human factors in computing systems. 643–652. [80] Frans Mäyrä. 2017. Dialogue and interaction in role-playing games: Playful communication as Ludic culture. In Dialogue across Media. John Benjamins Publishing Company, 271–290. [81] Anna McCoy, Jennifer Holloway, Olive Healy, Mandy Rispoli, and Leslie Neely. 2016. A systematic review and evaluation of video modeling, role-play and computer-based instruction as social skills interventions for children and adolescents with high-functioning autism. Review Journal of Autism and Developmental Disorders 3 (2016), 48–67. [82] Haylie L Miller and Nicoleta L Bugnariu. 2016. Level of immersion in virtual environments impacts the ability to assess and teach social skills in autism spectrum disorder. Cyberpsychology, Behavior, and Social Networking 19, 4 (2016), 246–256. [83] Damian EM Milton. 2012. On the ontological status of autism: The ‘double empathy problem’. Disability & society 27, 6 (2012), 883–887. [84] OpenAI. 2025. DALL·E 3. https://openai.com/index/dall-e-3/. Accessed: 202504-07. [85] OpenAI. 2025. GPT-4. https://openai.com/index/gpt-4/. Accessed: 2025-04-07. [86] Oracle. 2025. MySQL. https://www.mysql.com/. Accessed: 2025-04-07. [87] Pallets. 2010. Flask. https://flask.palletsprojects.com/en/stable/. Accessed: 2025-04-07. [88] Juhee Park. 2019. A Comparison of the Pretending Elements between Constructive Play and Pretend Play. Turkish Online Journal of Educational TechnologyTOJET 18, 4 (2019), 1–6. [89] Kaśka Porayska-Pomsta, Alyssa M. Alcorn, Katerina Avramides, Sandra Beale, Sara Bernardini, Mary Ellen Foster, Christopher Frauenberger, Judith Good, Karen Guldberg, Wendy Keay-Bright, Lila Kossyvaki, Oliver Lemon, Marilena Mademtzi, Rachel Menzies, Helen Pain, Gnanathusharan Rajendran, Annalu Waller, Sam Wass, and Tim J. Smith. 2018. Blending Human and Artificial Intelligence to Support Autistic Children’s Social Communication Skills. ACM Trans. Comput.-Hum. Interact. 25, 6, Article 35 (dec 2018), 35 pages. https: //doi.org/10.1145/3271484 [90] Linda M Quirmbach, Alan J Lincoln, Monica J Feinberg-Gizzo, Brooke R Ingersoll, and Siri M Andrews. 2009. Social stories: Mechanisms of effectiveness in increasing game play skills in children diagnosed with autism spectrum disorder using a pretest posttest repeated measures randomized control group design. Journal of autism and developmental disorders 39 (2009), 299–321. [91] Remove.bg. 2025. Remove.bg. https://www.remove.bg/. Accessed: 2025-04-07. [92] Julien G Rosselet and Sarah D Stauffer. 2013. Using group role-playing games with gifted children and adolescents: A psychosocial intervention model. International Journal of Play Therapy 22, 4 (2013), 173. [93] ZH Sain, A Asfahani, and N Krisnawati. 2022. Utiliziation AI for Socially Responsive Education as a Path to Inclusive Development. Journal of Artificial Intelligence and Development 1, 2 (2022), 69–78. [94] Brian Scassellati, Laura Boccanfuso, Chien-Ming Huang, Marilena Mademtzi, Meiying Qin, Nicole Salomons, Pamela Ventola, and Frederick Shic. 2018. Improving social skills in children with ASD using a long-term, in-home social robot. Science Robotics 3, 21 (2018), eaat7544. [95] Neelam Kharod Sell, Ellen Giarelli, Nathan Blum, Alexandra L Hanlon, and Susan E Levy. 2012. A comparison of autism spectrum disorder DSM-IV criteria and associated features among African American and white children in Philadelphia County. Disability and Health Journal 5, 1 (2012), 9–17. [96] Marsha Mailick Seltzer, Marty Wyngaarden Krauss, Paul T Shattuck, Gael Orsmond, April Swe, and Catherine Lord. 2003. The symptoms of autism spectrum disorders in adolescence and adulthood. Journal of autism and developmental disorders 33 (2003), 565–581. [97] Smita Shukla-Mehta, Trube Miller, and Kevin J Callahan. 2010. Evaluating the effectiveness of video instruction on social and communication skills training for children with autism spectrum disorders: A review of the literature. Focus

CHI ’26, April 13–17, 2026, Barcelona, Spain

on Autism and Other Developmental Disabilities 25, 1 (2010), 23–36. [98] GovTech Singapore. 2020. C.O.S.T.A.R. Framework for AI Prompt Engineering. https://www.tech.gov.sg. Accessed: 2024-09-07. [99] SS Sparrow, DV Cicchetti, and DA Balla. 2005. Vineland adaptive behavior scales–Second edition (Vineland–II). Circle Pines, MN: American Guidance Service (2005). [100] Ratna Suryani, Sugiyo Pranoto, and Budi Astuti. 2020. The effectiveness of storytelling and roleplaying media in enhancing early childhood empathy. Journal of Primary Education 9, 5 (2020), 546–553. [101] Yilin Tang, Liuqing Chen, Ziyu Chen, Wenkai Chen, Yu Cai, Yao Du, Fan Yang, and Lingyun Sun. 2024. EmoEden: Applying Generative Artificial Intelligence to Emotional Learning for Children with High-Function Autism. In Proceedings of the CHI Conference on Human Factors in Computing Systems. 1–20. [102] Andrea Tartaro, Justine Cassell, Corina Ratz, Jennifer Lira, and Valeria NanclaresNogués. 2014. Accessing peer social interaction: using authorable virtual peer technology as a component of a group social skills intervention program. ACM Transactions on Accessible Computing (TACCESS) 6, 1 (2014), 1–29. [103] Unity Technologies. 2025. Unity. https://unity.com/cn. Accessed: 2025-04-07. [104] Caitlin Tenison, Jon M Fincham, and John R Anderson. 2016. Phases of learning: How skill acquisition impacts cognitive processing. Cognitive psychology 87 (2016), 1–28. [105] John Terry, Gerald Strait, Steve Alsarraf, Emily Weinmann, and Allison Waychoff. 2025. Artificial intelligence in scale development: evaluating AI-generated survey items against gold standard measures. Current Psychology (2025), 1–12. [106] Céliane Trudel and Aparna Nadig. 2019. A role-play assessment tool and drama-based social skills intervention for adults with autism or related social communication difficulties. Dramatherapy 40, 1 (2019), 41–60. [107] Wei-Te Tsai, I-Jui Lee, and Chien-Hsu Chen. 2021. Inclusion of third-person perspective in CAVE-like immersive 3D virtual reality role-playing games for social reciprocity training of children with an autism spectrum disorder. Universal Access in the Information Society 20, 2 (2021), 375–389. [108] Steffie Van Der Steen, Carla H Geveke, Anne T Steenbakkers, and Henderien W Steenbeek. 2020. Teaching students with autism spectrum disorders: What are the needs of educational professionals? Teaching and Teacher Education 90 (2020), 103036. [109] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022. Chain of thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems. [110] Susan Williams White, Kathleen Keonig, and Lawrence Scahill. 2007. Social skills development in children with autism spectrum disorders: A review of the intervention research. Journal of autism and developmental disorders 37 (2007), 1858–1868. [111] Jiazhou Wu, Min Fan, Liyan Sheng, and Guoyu Sun. 2023. Exploring the design space of virtual tutors for children with autism spectrum disorder. Education and Information Technologies 28, 12 (2023), 16531–16560. [112] Evan You. 2024. Vue.js. https://v2.vuejs.org/. Accessed: 2025-04-07. [113] Chao Zhang, Cheng Yao, Jiayi Wu, Weijia Lin, Lijuan Liu, Ge Yan, and Fangtian Ying. 2022. StoryDrawer: a child–AI collaborative drawing system to support children’s creative visual storytelling. In Proceedings of the 2022 CHI conference on human factors in computing systems. 1–15. [114] Cheng Zheng, Caowei Zhang, Xuan Li, Fan Zhang, Bing Li, Chuqi Tang, Cheng Yao, Ting Zhang, and Fangtian Ying. 2017. KinToon: a kinect facial projector for communication enhancement for ASD children. In Adjunct Proceedings of the 30th Annual ACM Symposium on User Interface Software and Technology. 201–203.

A The prompts for visual generation. A.1 Universal Rules for Visual Generation

The prompts for visual generation consist of two components: universal rules and user-defined inputs. The universal rules, which ensure consistency across all generations, are defined as follows:

(1) Child Character. The prompt generates a cute, bright 2D cartoon-style visual, suitable for kids, based on age, sex, and appearance, with a simple white background to keep the focus on the character. (2) Teacher Character. Similar to the child character prompt structure, the teacher visual is generated with specified age, sex, and appearance, maintaining consistency in style and presentation.

Li et al.

(3) Background Scenes. The prompt generates a scene based on the selected description, using a bright, minimalist, cartoonlike style to support the narrative without distractions. (4) Reinforcer Icons. The prompt is to produce an icon representing a specific reinforcement text. The icon is bright, cartoon-like, isolated from any characters, focusing entirely on the reinforcer element.

A.2 User-defined Input and Corresponding Images in GenRole

Table 5 then illustrates an example of user-defined inputs along with the corresponding visuals generated by DALL·E 3.

GenRole: Personalizing Role Play for Educators Supporting Autistic Students’ Social Interaction Learning

CHI ’26, April 13–17, 2026, Barcelona, Spain

Table 5: User-defined input and corresponding images in GenRole.

Type

User-Defined Input

Child Character

<8 years old> <boy> <short black hair, wearing a red cap, summer clothes, and sneakers>

Teacher Character

<55 years old> <man> <bald, beer belly, wearing a suit and leather shoes>

Background Scenes

An empty classroom with a blackboard in the front and tables and desks.

Reinforcer Icons

a furry toy rabbit

Example Image
