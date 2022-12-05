import streamlit as st
import streamlit.components.v1 as components


def main():

    st.markdown("# Tableau Public Dashboard")
    st.markdown('''##### The dataset used for the dashboard and training was originally collected by CDCs Behavioral Risk Factor Surveillance System (BRFSS) which conducts surveys and collects data on the health status of US residents. Initially there were 401,958 instances with 279 columns. Most columns were questions that were asked to respondents such as "Do you have serious difficulty walking or climbing stairs?" or "Have you smoked at least 100 cigarettes in your entire life?”. The author of this data has cleaned this data and selected relevant variables that can be used to detect early heart failures.''')

    st.markdown("*For best usability, please collapse the sidebar if open*")

    tableau_public = '''<div class='tableauPlaceholder' id='viz1670173916615' style='position: relative'><noscript><a href='#'><img alt='Heart Diseases ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;He&#47;HeartDiseases_16682166124340&#47;HeartDiseases&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='HeartDiseases_16682166124340&#47;HeartDiseases' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;He&#47;HeartDiseases_16682166124340&#47;HeartDiseases&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1670173916615');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.height='1177px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>'''
    components.html(tableau_public, height=1650, scrolling=False)

if __name__ == '__main__':
    main()