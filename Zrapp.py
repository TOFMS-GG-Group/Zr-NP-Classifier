import streamlit as st
from helper import *
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_theme()
import plotly.express as px
import xarray as xr

st.title("Zirconium Particle Classification")
st.write("---" * 134)

''


@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')


with st.sidebar:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 300px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 300px;
            margin-left: -300px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("File upload")
    file_type = st.radio(
        "Select upload file type",
        ('CSV', 'Excel'))
    if file_type == "Excel":
        tabname = st.text_input('Enter excel sheet name')
        st.session_state["uploaded_file"] = "init_xlsx"
        if tabname:
            st.session_state["uploaded_file"] = "xlsx"
            uploaded_file = st.file_uploader("Upload CSV/XLSX file", type=['xlsx'], accept_multiple_files=False)

    elif file_type == "CSV":
        st.session_state["uploaded_file"] = "CSV"
        tabname = st.text_input('FileNameDescription')
        uploaded_file = st.file_uploader("Upload CSV/XLSX file", type=['CSV'], accept_multiple_files=False)

    # to handle session and UI
    if st.session_state["uploaded_file"] == "xlsx" and uploaded_file is not None:
        file_details = {"name": uploaded_file.name, "type": uploaded_file.type, "proceed": True}
    elif st.session_state["uploaded_file"] == "CSV" and uploaded_file is not None:
        file_details = {"name": uploaded_file.name, "type": uploaded_file.type, "proceed": True}
    else:
        file_details = {"proceed": False}

    if uploaded_file:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(uploaded_file)
        #st.subheader("spTime and qPlasma")
        # Assuming the element name is in the second column (index 1)
        element_column_index = 1
        spTime = df.at[df.index[df['Element'] == 'spTime (s)'].tolist()[0], df.columns[element_column_index]]
        #st.write(f"spTime: {spTime}")
        qPlasma = df.at[df.index[df['Element'] == 'qPlasma (mL/s)'].tolist()[0],df.columns[element_column_index]]
        #st.write(f"qPlasma: {qPlasma}")

        Lc_row_name = 'Avg Lc (TofCts)'
        Sensitivity_row_name = 'Sensitivity (TofCts/g)'
        Sdrop_Zr1 = df.at[df.index[df['Element'] == Sensitivity_row_name].tolist()[0], 'Zr']
        # st.write(f"Sdrop of Zr1: {Sdrop_Zr1}")
        Sdrop_Zr = np.float64(Sdrop_Zr1) / (10 ** 18)  ##(in terms of cts/ag)
        # st.write(f"Sdrop of Zr: {Sdrop_Zr}")



def classifiers_Zr(dataframe):
    start_index = 12  # Index of the row containing "Particle Signals (TofCts)"
    data = dataframe.iloc[start_index:]
    result_classify = []
    columns_to_check_Zr = [column for column in data.columns if column != "Zr"
                           and column != "Hf" and column != data.columns[0]]
    for index, row in data.iterrows():
        Zr_value = float(row["Zr"]) / float(Sdrop_Zr)
        all_zero_Zr = all(float(row[column]) == 0 for column in columns_to_check_Zr)

        if float(row["Zr"]) > 0:
            if (float(row["Hf"]) > 0) and all_zero_Zr:
                                result_classify.append("Zr-eng")
            elif (float(row["Zr"]) > 0) and all_zero_Zr:
                                result_classify.append("unc sm-Zr")
            elif (float(row["Zr"]) > 0):
                if all_zero_Zr is False:
                    result_classify.append("Zr-nat")
                else:
                    result_classify.append("unc mm-Zr")
            else:
                result_classify.append("unclassified-Zr")
        else:
            result_classify.append("non Zr NPs")
    return pd.DataFrame(result_classify, columns=["classification_Zr"])



if file_details["proceed"]:
    try:
        print(file_details)
        saved_file = savefileindrive(uploaded_file)
        if saved_file == 1:
            print(f"Reading...{file_details['name']}..{tabname}")
            try:
                if uploaded_file.name.split(".")[1] == "csv":
                    dataframe = pd.read_csv(f"data/{file_details['name']}")
                elif uploaded_file.name.split(".")[1] == "xlsx":
                    dataframe = pd.read_excel(f"data/{file_details['name']}", sheet_name=f"{tabname}")
                st.subheader("Original data file")

                st.write(dataframe)
                csvresult = convert_df(dataframe)
                # Create a copy of the dataframe
                dataframe_mass = dataframe.copy()

                # Retrieve the sensitivity row and clean it by removing non-numeric characters and scientific notation
                sensitivity_row = dataframe_mass.iloc[1, 1:]  # Assuming sensitivity row is at index 2
                sensitivity_row = sensitivity_row.str.replace('[^\d.]', '')  # Remove non-numeric characters
                sensitivity_row = pd.to_numeric(sensitivity_row, errors='coerce') / (10 ** 18)

                # Filter out zero values in the sensitivity row
                non_zero_sensitivity_row = sensitivity_row[sensitivity_row != 0]
                # st.write(non_zero_sensitivity_row)

                # Align the sensitivity row with the corresponding columns in the DataFrame
                non_zero_sensitivity_row = non_zero_sensitivity_row.reindex(dataframe_mass.columns[1:])

                # Convert the relevant columns in classified_combined to numeric data type
                dataframe_mass.iloc[12:, 1:] = dataframe_mass.iloc[12:, 1:].apply(pd.to_numeric, errors='coerce')

                # Divide the rows below indexing starts by the sensitivity row
                #st.subheader("Classified results in MASS: HIDE THIS LATER")
                dataframe_mass.iloc[12:, 1:] = dataframe_mass.iloc[12:, 1:].div(non_zero_sensitivity_row.values, axis=1)
                #st.write(dataframe_mass)

                classified_combined = dataframe_mass.copy()

                # Call the classifiers_Zr function and store the classified results
                classifiedres_Zr = classifiers_Zr(dataframe)

                # Update the classified_combined dataframe for Zr classification
                classified_combined["classification_Zr"] = ""  # Add an empty column for classification
                classified_combined.loc[12:, "classification_Zr"] = classifiedres_Zr[
                                                                    :len(classified_combined) - 12].values


                # Save the classified data to a CSV file
                csvresult = convert_df(classified_combined)

                # Convert the relevant columns in classified_combined to numeric data type
                classified_combined.iloc[12:, 1:-3] = classified_combined.iloc[12:, 1:-3].apply(pd.to_numeric,
                                                                                              errors='coerce')




                # Add information to the second column and 12th row
                information_text = "Mass amounts in Attograms"
                classified_combined.at[10, classified_combined.columns[0]] = information_text

                # Save the classified data to a CSV file
                csvresult = convert_df(classified_combined)

                # Divide the rows below indexing starts by the sensitivity row
                st.subheader("Classified results in attograms")
                st.write(classified_combined)

                # Divide the rows below indexing starts by the sensitivity row
                st.subheader("Classified results in attograms")
                #######Zirconium classification#######
                Zrenp = classified_combined[classified_combined["classification_Zr"] == "Zr-eng"][
                    "classification_Zr"].count()
                Zrnat = classified_combined[classified_combined["classification_Zr"] == "Zr-nat"][
                    "classification_Zr"].count()
                uncsmZr = classified_combined[classified_combined["classification_Zr"] == "unc sm-Zr"][
                    "classification_Zr"].count()
                unclassifiedZr = classified_combined[classified_combined["classification_Zr"] == "unclassified-Zr"][
                    "classification_Zr"].count()
                unclassifiedmmZr = \
                    classified_combined[classified_combined["classification_Zr"] == "unc mm-Zr"][
                        "classification_Zr"].count()
                nonZrNPs = classified_combined[classified_combined["classification_Zr"] == "non Zr NPs"][
                    "classification_Zr"].count()

                st.download_button(
                    label="Download result",
                    data=csvresult,
                    file_name='large_df.csv',
                    mime='text/csv',)

                ##################----Table----Nbr and PNCs-----#################
                st.write("Table of Nbr of Particles and PNCs")


                # Function to create a table
                def create_table(data_dict, spTime, qPlasma):
                    particle_names = list(data_dict.keys())
                    particle_counts = list(data_dict.values())
                    # Convert spTime and qPlasma to numeric types
                    spTime = float(spTime)
                    qPlasma = float(qPlasma)

                    # Calculate PNCs for each particle
                    pncs_values = [int(count / (spTime * qPlasma) if (spTime * qPlasma) != 0 else 0) for count in
                                   particle_counts]

                    table_data = {
                        'Particle Name': particle_names,
                        'Number of Particles': particle_counts,
                        'PNCs of Particles': pncs_values
                    }

                    return pd.DataFrame(table_data)

                zr_particles = {
                    'Zr-eng': Zrenp,
                    'Zr-nat': Zrnat,
                    'unc sm-Zr': uncsmZr,
                    'unc mm-Zr': unclassifiedmmZr,
                    'non Zr NPs': nonZrNPs,
                    'unclassified-Zr': unclassifiedZr
                }
                spTime_value = spTime
                qPlasma_value = qPlasma

                # Display tables for each category
                st.subheader("Zr-containing Particles")
                st.table(create_table(zr_particles, spTime_value, qPlasma_value))

                ######Pie Charts##################
                st.write("Zr-particles")
                labels = ['unclassified-Zr', 'Zr-nat', 'Zr-eng', 'unc sm-Zr', 'unc mm-Zr', 'non Zr Nps']
                sizes = [unclassifiedZr, Zrnat, Zrenp, uncsmZr, unclassifiedmmZr, nonZrNPs]

                fig = go.Figure(data=[go.Pie(labels=labels, values=sizes, textinfo='label+percent+value',
                                             insidetextorientation='radial', title='Zr and non-Zr particles',
                                             )])

                fig

                st.write("Zr-Only particles")
                labels = ['unclassified-Zr', 'Zr-nat', 'Zr-eng', 'unc sm-Zr', 'unc mm-Zr']
                sizes = [unclassifiedZr, Zrnat, Zrenp, uncsmZr, unclassifiedmmZr]

                fig = go.Figure(data=[go.Pie(labels=labels, values=sizes, textinfo='label+percent+value',
                                             insidetextorientation='radial', title='Zr only particles',
                                             )])
                fig


                ##########calculating the total mass of classified particles#########
                Zr_nat_rows = classified_combined[classified_combined['classification_Zr'] == 'Zr-nat']
                st.write("Zr-mass only in Zr_nat particles")
                Zr_nat_mass = Zr_nat_rows['Zr'].sum()
                st.write(int(Zr_nat_mass))

                st.write("Total mass of each element in Zr_nat particles")
                columns_to_sum = Zr_nat_rows.columns[1:-3]
                Zr_nat_mass_total = Zr_nat_rows[columns_to_sum].astype(float).sum()
                Zr_nat_mass_total_rounded = round(Zr_nat_mass_total, 0)
                st.write(Zr_nat_mass_total_rounded)

                Zr_eng_rows = classified_combined[classified_combined['classification_Zr'] == 'Zr-eng']
                Zr_eng_mass = Zr_eng_rows['Zr'].sum()
                st.write("Zr-mass only in Zr_eng particles")
                st.write(int(Zr_eng_mass))


                st.write("Density Heat map in terms of mass (ag)")
                mass_cols = classified_combined.iloc[12:, 1:-3]
                # Exclude 46Ti and 48Ti columns
                columns_to_exclude = ['46Ti', '48Ti']
                mass_cols = mass_cols.drop(columns=columns_to_exclude)
                fig1 = px.imshow(mass_cols, origin='upper', zmin=0,
                                 zmax=100000)

                fig['layout'].update(paper_bgcolor='salmon', plot_bgcolor='salmon')
                fig1['layout'].update(plot_bgcolor='#FFFFFF')

                fig.update_layout(plot_bgcolor="salmon")
                # fig1.update_layout(plot_bgcolor='black')
                fig1.update_layout(width=600, height=400, margin=dict(l=50, r=10, b=30, t=10))
                fig1.update_xaxes(showticklabels=True).update_yaxes(showticklabels=True)
                st.plotly_chart(fig1)

                ###########___Correlation_Matrix_____############
                st.write("Correlation Matrix plot")

                # Assuming classified_combined is your DataFrame
                selected_cols = classified_combined.iloc[12:, 1:-1]

                # Convert string values to numeric (remove commas and convert to float)
                selected_cols = selected_cols.replace(',', '', regex=True).astype(float)

                # Exclude the columns you want to ignore (e.g., '46Ti' and '48Ti')
                columns_to_exclude = ['46Ti']
                selected_cols_filtered = selected_cols.drop(columns=columns_to_exclude)

                # Calculate correlation matrix
                corr_matrix = selected_cols_filtered.corr()

                # Create a scatter plot matrix with bubble sizes based on correlation values
                plt.figure(figsize=(14, 10))
                sns.set_theme(context="notebook", style="whitegrid", palette="deep", font="sans-serif", font_scale=1.2,
                              color_codes=True, rc=None)

                # Use seaborn's heatmap to display the correlation matrix
                sns.heatmap(corr_matrix, annot=True, cmap='viridis', vmin=-1, vmax=1, fmt=".2f")

                # Show the plot
                plt.xticks(rotation=90)
                plt.yticks(rotation=0)
                plt.show()
                st.pyplot(plt)

                ###############Correlation between 2 elements#############

                # Assuming 'classified_combined' is your DataFrame
                st.write("Scatter Plot of selected elements in selected particle type")
                from scipy.stats import linregress
                from matplotlib.ticker import MaxNLocator, ScalarFormatter, EngFormatter

                # Get the unique elements for the dropdown menu
                element_options = classified_combined.columns.tolist()
                particle_options_Zr = ['unclassified-Zr', 'Zr-nat', 'Zr-eng',
                                       'unc sm-Zr', 'unc mm-Zr', 'non Zr Nps']


                # Default selected elements
                default_element_Ti = 'Ti_4648_corr'
                default_element_Zr = 'Zr'
                default_element_Nb = 'Nb'
                default_element_Ce = 'Ce'
                default_element_La = 'La'
                default_element_Hf = 'Hf'
                default_particle_type = 'Zr'
                default_particle = 'Zr-eng'

                # Dropdowns for particle type selection
                selected_particle_type = st.selectbox("Select Particle Type", ['Ti', 'Zr', 'Ce'],
                                                      index=['Ti', 'Zr', 'Ce'].index(default_particle_type))

                # Select particle options based on the particle type

                if selected_particle_type == 'Zr':
                    particle_options = particle_options_Zr
                    default_particle = 'Zr-nat'
                    selected_element_1 = st.selectbox("Select Element 1", element_options,
                                                      index=element_options.index(default_element_Zr))
                    selected_element_2 = st.selectbox("Select Element 2", element_options,
                                                      index=element_options.index(default_element_Hf))

                # Dropdown for particle selection
                selected_particle = st.selectbox(f"Select {selected_particle_type}-Particle", particle_options,
                                                 index=particle_options.index(default_particle))

                # Filter data for the selected particle
                filtered_data = classified_combined[
                    classified_combined[f'classification_{selected_particle_type}'] == selected_particle]

                # Drop rows with NaN values in selected elements
                filtered_data = filtered_data.dropna(subset=[selected_element_1, selected_element_2])

                # Convert to numeric
                filtered_data[selected_element_1] = pd.to_numeric(filtered_data[selected_element_1], errors='coerce')
                filtered_data[selected_element_2] = pd.to_numeric(filtered_data[selected_element_2], errors='coerce')

                # Scatter plot using Matplotlib
                fig, ax = plt.subplots()

                # Filter data for the selected particle and non-zero values
                filtered_data_non_zero = filtered_data[
                    (filtered_data[selected_element_1] != 0) & (filtered_data[selected_element_2] != 0)]

                scatter = ax.scatter(
                    x=filtered_data_non_zero[selected_element_1],
                    y=filtered_data_non_zero[selected_element_2],
                )
                # Perform linear regression
                slope, intercept, r_value, p_value, std_err = linregress(filtered_data[selected_element_1],
                                                                         filtered_data[selected_element_2])
                # Add trendline to the plot
                trendline_x = np.linspace(min(filtered_data[selected_element_1]),
                                          max(filtered_data[selected_element_1]), 100)
                trendline_y = slope * trendline_x + intercept
                ax.plot(trendline_x, trendline_y, color='red', label=f'Regression Line: {slope:.4f}x + {intercept:.4f}')

                # Customize plot labels and title
                ax.set_title(f"Scatter Plot for {selected_particle_type}-{selected_particle}")
                ax.set_xlabel(f"{selected_element_1} (mass in ag)")
                ax.xaxis.set_major_formatter(EngFormatter(useMathText=True))
                ax.set_ylabel(f"{selected_element_2} (mass in ag)")
                ax.yaxis.set_major_formatter(EngFormatter(useMathText=True))

                # Annotate the plot with equation and R-squared value
                equation_text = f"Equation: y = {slope:.4f}x + {intercept:.4f}"
                r_squared_text = f"R-squared: {r_value ** 2:.4f}"
                ax.annotate(equation_text, xy=(0.1, 0.95), xycoords='axes fraction', fontsize=10)
                ax.annotate(r_squared_text, xy=(0.1, 0.9), xycoords='axes fraction', fontsize=10)

                # Show the plot
                st.pyplot(fig)

                ################################------HISTOGRAMS-----#########################
                st.write("Histogram of select element mass")
                import matplotlib.pyplot as plt
                from matplotlib.ticker import MaxNLocator, ScalarFormatter, EngFormatter

                ############################--------Titanium_Particles-------#################
                # Get the unique elements for the dropdown menu
                Particle_options = ['unclassified-Zr', 'Zr-nat', 'Zr-eng',
                                       'unc sm-Zr', 'unc mm-Zr', 'non Zr Nps']

                element_options = classified_combined.columns[:]  # Assuming the elements start from the second column

                # Default selected element
                default_element = 'Zr'
                default_particle = 'Zr-eng'

                # Create a dropdown menu for selecting the element
                selected_element = st.selectbox("Select Element", element_options,
                                                index=element_options.get_loc(default_element))
                selected_particle = st.selectbox("Select Particle", Particle_options,
                                                 index=Particle_options.index(default_particle))

                # Assuming 'classification_Ti' is the column indicating rutile classification
                Zr_particles = classified_combined[classified_combined['classification_Zr'] == selected_particle]

                # Plot histogram for the selected element in rutile particles
                fig, ax = plt.subplots()
                sns.histplot(Zr_particles[selected_element], bins=20, kde=False, ax=ax)

                # Set x-axis to log scale
                # ax.set_xscale('log')
                ax.set_yscale('log')

                # Automatically set 10 ticks on the x-axis
                ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
                # Use engineering notation for the x-axis labels
                ax.xaxis.set_major_formatter(EngFormatter(useMathText=True))
                # Set font size for x-axis ticks
                ax.tick_params(axis='x', labelsize=10)

                # Automatically set 10 ticks on the y-axis
                ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
                # Use engineering notation for the y-axis labels
                ax.yaxis.set_major_formatter(EngFormatter(useMathText=True))
                # Set font size for y-axis ticks
                ax.tick_params(axis='y', labelsize=10)

                # Set x-axis, y-axis and tile label
                ax.set_xlabel('Mass (ag)', fontsize=12)
                ax.set_ylabel('Count', fontsize=12)
                ax.set_title('Histogram of select element mass in Zirconium Particle', fontsize=15)
                # Show the plot in Streamlit
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error reading the file: {e}")
    except Exception as e:
        st.error(f"Error processing the file: {e}")










