import pandas as pd
import numpy as np

# Load original data to keep the structure
df_original = pd.read_csv('computer_science_student_career_datasetMar62024.csv')

# 1. Drop the requested columns
# Added 'Others' to the list as requested
cols_to_drop = ['GPA', 'Coursework_Completion_Status', 'Academic_Achievements', 'Others']
df_original = df_original.drop(columns=[c for c in cols_to_drop if c in df_original.columns])

# 2. Fix Personal_Interests (remove 'CodingOthers')
# Replace 'CodingOthers' with 'Coding' or other relevant interests
if 'Personal_Interests' in df_original.columns:
    df_original['Personal_Interests'] = df_original['Personal_Interests'].replace('CodingOthers', 'Coding')

all_tracks = [
    'ai-agents', 'ai-data-scientist', 'ai-engineer', 'ai-red-teaming', 'android',
    'angular', 'aspnet-core', 'aws', 'backend', 'blockchain', 'cloudflare',
    'computer-science', 'cpp', 'cyber-security', 'data-analyst', 'devops',
    'devrel', 'docker', 'engineering-manager', 'frontend', 'full-stack',
    'game-developer', 'kubernetes', 'linux', 'mlops', 'mongodb', 'nodejs',
    'php', 'product-manager', 'python', 'qa', 'react', 'react-native', 'redis',
    'rust', 'sql', 'system-design', 'technical-writer', 'terraform',
    'typescript', 'ux-design', 'vue'
]

# Define "Signature Skills" for each track
signatures = {
    'ai-agents': ['Python', 'Problem_Solving_Abilities'],
    'ai-data-scientist': ['Python', 'Database_Management', 'Problem_Solving_Abilities'],
    'ai-engineer': ['Python', 'Software_Development_Experience'],
    'ai-red-teaming': ['Networking_Skills', 'Problem_Solving_Abilities', 'Python'],
    'android': ['Java', 'Software_Development_Experience', 'Adaptability'],
    'angular': ['JavaScript', 'Web_Development_Experience', 'TypeScript'],
    'aspnet-core': ['C#', 'Software_Development_Experience', 'Database_Management'],
    'aws': ['Networking_Skills', 'Adaptability'],
    'backend': ['Python', 'Database_Management', 'Software_Development_Experience'],
    'blockchain': ['Go', 'C++', 'Problem_Solving_Abilities'],
    'cloudflare': ['Networking_Skills', 'JavaScript', 'Adaptability'],
    'computer-science': ['C++', 'Java', 'Problem_Solving_Abilities'],
    'cpp': ['C++', 'Software_Development_Experience', 'Problem_Solving_Abilities'],
    'cyber-security': ['Networking_Skills', 'Problem_Solving_Abilities', 'Adaptability'],
    'data-analyst': ['Python', 'Database_Management', 'Time_Management'],
    'devops': ['Networking_Skills', 'Software_Development_Experience', 'Time_Management'],
    'devrel': ['Communication_Skills', 'Teamwork_Collaboration'],
    'docker': ['Networking_Skills', 'Software_Development_Experience', 'Adaptability'],
    'engineering-manager': ['Communication_Skills', 'Teamwork_Collaboration', 'Leadership_Experience'],
    'frontend': ['JavaScript', 'Web_Development_Experience'],
    'full-stack': ['JavaScript', 'Python', 'Web_Development_Experience', 'Database_Management'],
    'game-developer': ['C++', 'Problem_Solving_Abilities'],
    'kubernetes': ['Networking_Skills', 'Software_Development_Experience', 'Adaptability'],
    'linux': ['Networking_Skills', 'Problem_Solving_Abilities'],
    'mlops': ['Python', 'Networking_Skills', 'Software_Development_Experience'],
    'mongodb': ['Database_Management', 'JavaScript'],
    'nodejs': ['JavaScript', 'Software_Development_Experience', 'Database_Management'],
    'php': ['PHP', 'Web_Development_Experience', 'Database_Management'],
    'product-manager': ['Communication_Skills', 'Time_Management', 'Leadership_Experience'],
    'python': ['Python', 'Software_Development_Experience', 'Problem_Solving_Abilities'],
    'qa': ['Problem_Solving_Abilities', 'Time_Management'],
    'react': ['JavaScript', 'Web_Development_Experience', 'Adaptability'],
    'react-native': ['JavaScript', 'Software_Development_Experience', 'Adaptability'],
    'redis': ['Database_Management', 'Time_Management'],
    'rust': ['Rust', 'Software_Development_Experience', 'Problem_Solving_Abilities'],
    'sql': ['Database_Management', 'Time_Management'],
    'system-design': ['Software_Development_Experience', 'Networking_Skills', 'Problem_Solving_Abilities'],
    'technical-writer': ['Communication_Skills', 'Time_Management'],
    'terraform': ['Networking_Skills', 'Software_Development_Experience', 'Adaptability'],
    'typescript': ['JavaScript', 'Web_Development_Experience'],
    'ux-design': ['Adaptability', 'Communication_Skills'],
    'vue': ['JavaScript', 'Web_Development_Experience', 'Adaptability']
}

def generate_row(track):
    # Start with low background noise (1-4)
    row = {col: np.random.randint(1, 5) for col in df_original.columns if col not in ['Personal_Interests', 'Internship_Experience', 'Certifications_Training', 'Leadership_Experience', 'Career_Goals']}
    
    # Apply high signature skills (7-9)
    sig_skills = signatures.get(track, [])
    for skill in sig_skills:
        if skill in row:
            row[skill] = np.random.randint(7, 10)
    
    # Categorical
    row['Personal_Interests'] = np.random.choice(['Coding', 'Research', 'Design', 'Management'])
    row['Internship_Experience'] = np.random.choice(['Yes', 'No'])
    row['Certifications_Training'] = np.random.choice(['Yes', 'No'])
    row['Leadership_Experience'] = 'Yes' if 'Leadership_Experience' in sig_skills or track in ['product-manager', 'engineering-manager'] else 'No'
    
    row['Career_Goals'] = track
    return row

# Create fully synthetic balanced dataset
target_count = 2000
balanced_data = []

for track in all_tracks:
    synthetic_rows = [generate_row(track) for _ in range(target_count)]
    balanced_data.append(pd.DataFrame(synthetic_rows))

df_final = pd.concat(balanced_data, ignore_index=True)
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

df_final.to_csv('cleaned_career_dataset.csv', index=False)
print(f"Dataset rebuilt. Removed 'Others' and fixed 'Personal_Interests'. Total rows: {len(df_final)}")
