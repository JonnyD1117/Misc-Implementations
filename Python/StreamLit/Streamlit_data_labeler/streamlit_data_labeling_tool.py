import streamlit as st
import os

input_path = "./unlabled_data/"
output_path = "./labled_data/"

img_list = os.listdir(input_path)

st.title("Streamlit Data Labeler App")
image_state = 0
col1, col2, col3 = st.beta_columns([1, 4, 1])

# box_select_state = st.sidebar.selectbox('Please Select Something', ['Label Images', 'Option Two'])

my_bar = st.progress(0)


@st.cache
def image_path_func(img_counter):
    return input_path + img_list[img_counter]


progress_bar_state = 0

if __name__ == "__main__":

    if st.sidebar.selectbox('Please Select Something', ['Label Images', 'Option Two']) == 'Label Images':
        img_path = image_path_func(image_state)
        # labeling_date = st.date_input("Enter Date & Time")

        with col1:
            st.header("Prev.")

            prev_btn_state = st.button("Previous")

            if prev_btn_state is True and image_state == 0:
                st.warning("Beginning of Input Image List")

            elif prev_btn_state is True and image_state > 0:
                image_state -= 1
                img_path = image_path_func(image_state)

        with col2:
            st.header(f"Label This Image:  {image_state}  of {len(img_list)}")
            st.image(img_path, use_column_width=True)
            # my_bar.progress(progress_bar_state)

        with col3:
            st.header("Next" )

            next_btn_state = st.button("Next",)

            if next_btn_state is True and image_state == len(img_list):
                st.warning("End of Input Image List")

            elif next_btn_state is True and image_state < len(img_list):
                progress_bar_state += 10
                # my_bar.progress(progress_bar_state)
                image_state = image_state + 1
                img_path = image_path_func(image_state)
                print(image_state)

        my_bar.progress(progress_bar_state)


    else:
        st.write("In ELSE")



    # st.file_uploader("Drop Additional Files Here")
