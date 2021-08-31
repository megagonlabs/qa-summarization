$(document).ready(function() {

    Set.prototype.intersection = function(otherSet)
    {
        // creating new set to store intersection
        var intersectionSet = new Set();
      
        // Iterate over the values 
        for(var elem of otherSet)
        {
            // if the other set contains a 
            // similar value as of value[i]
            // then add it to intersectionSet
            if(this.has(elem))
                intersectionSet.add(elem);
        }
      
    // return values of intersectionSet
    return intersectionSet;                
    }

    /* LCS */
    function max(a, b)
    {
        if (a > b)
            return a;
        else
            return b;
    }
     
    function lcs(X, Y, m, n)
    {
        var L = new Array(m + 1);
        for(var i = 0; i < L.length; i++)
        {
            L[i] = new Array(n + 1);
        }
        var i, j;
        for(i = 0; i <= m; i++)
        {
            for(j = 0; j <= n; j++)
            {
                if (i == 0 || j == 0)
                    L[i][j] = 0;
                else if (X[i - 1] == Y[j - 1])
                    L[i][j] = L[i - 1][j - 1] + 1;
                else
                    L[i][j] = max(L[i - 1][j], L[i][j - 1]);
            }
        }
        return L[m][n];
    }

    /* edit distance */
    function similarity(s1, s2) {
        var longer = s1;
        var shorter = s2;
        if (s1.length < s2.length) {
            longer = s2;
            shorter = s1;
        }
        var longerLength = longer.length;
        if (longerLength == 0) {
            return 1.0;
        }
        return (longerLength - editDistance(longer, shorter)) / parseFloat(longerLength);
    }

    function editDistance(s1, s2) {
        s1 = s1.toLowerCase();
        s2 = s2.toLowerCase();

        var costs = new Array();
        for (var i = 0; i <= s1.length; i++) {
            var lastValue = i;
            for (var j = 0; j <= s2.length; j++) {
                if (i == 0)
                    costs[j] = j;
                else {
                    if (j > 0) {
                        var newValue = costs[j - 1];
                        if (s1.charAt(i - 1) != s2.charAt(j - 1))
                            newValue = Math.min(Math.min(newValue, lastValue),
                                costs[j]) + 1;
                        costs[j - 1] = lastValue;
                        lastValue = newValue;
                    }
                }
            }
            if (i > 0)
                costs[s2.length] = lastValue;
        }
        return costs[s2.length];
    }

    var startTime, endTime;

    function start() {
        startTime = new Date();
    };

    function end() {
        endTime = new Date();
        var timeDiff = endTime - startTime; //in ms
        // strip the ms
        timeDiff /= 1000;

        // get seconds 
        var seconds = Math.round(timeDiff);
        // console.log(seconds + " seconds");
        return seconds
    }

    //localStorage.clear();
    
    var version = 1
    var index = 0
    var qa_info_list = []
    var check_submit = 0
    var click1=-1,click2=-1,click3=-1,click4=-1,click5=-1,click6=-1,click7=-1,click8=-1

    //annotation set
    // <replace_item_cat>
    // <replace_item_prod>
    // <replace_item_ques>
    // <replace_item_ans>

    var prod_cat_list =['Toys_and_Games', 'Toys_and_Games', 'Toys_and_Games', 'Toys_and_Games', 'Toys_and_Games', 'Toys_and_Games', 'Toys_and_Games', 'Toys_and_Games'] 
    var prod_idx_list =['B000FCURJC', 'B000FCURJC', 'B000FCURJC', 'B000FCURJC', 'B000FCURJC', 'B000FCURJC', 'B000FCURJC', 'B000FCURJC']
    var ques_list =['What are the dimensions of the puzzle space?', 'Is this actually a rigid board or more of a floppy mat?', 'will this mat hold 1000 piece puzzle?', 'will it hold a 24 x 30 inch puzzle?', 'how wide is each Side piece?', 'What size is the closed unit?', "So, if you have the center area for putting the puzzle together in, where do the pieces go that you haven't used yet when you close it up?", "Looking to purchase for an elderly person.  Can this be used in your lap while seated in a chair or sofa?  What is the weight?"]
    var ans_list =['THE DIMENSIONS FOR THE PUZZLE SPACE ARE 32" X 21.75"', "It's actually several rigid boards. You are able to arrange pieces in the middle and on two side pieces and then pick up those side pieces to place them atop the middle area before folding the wings in. It's really very cool.", 'Yes , it holds puzzles up to 1000 pieces .', 'May be a bit too small, according to the dimensions. Go with the 1500 size to be safe.', 'Each side piece is 16" wide and each insert (there are two) is 15-1/4 wide.  Hope this helps!', 'Closed is almost the same size as the puzzle workspace.  32.25 x 22', 'the 2 pieces fit on top of the puzzle when you close it up.', 'Its too big to use on your lap. Definitely needs a table.']
    
    var len = prod_idx_list.length
    console.log(len)

    // check porgress and restore
    //qa_info_list = JSON.parse(localStorage.getItem("data_info"));  
    //index = parseInt(localStorage.getItem("progress_index"));
    

    //if (qa_info_list == null){ qa_info_list = [] }
    //if (isNaN(index)){ index = 0 }

    // default data info
    for (i = 0; i < 8; i++){
        var q_txt = '.ques_txt_' + parseInt(i+1) + ' span'
        var a_txt = '.ans_txt_' + parseInt(i+1) + ' span'
        var qa_sum = '#Textarea' + parseInt(i+1)
        $(q_txt).text(ques_list[i])
        $(a_txt).text(ans_list[i])
        $(qa_sum).text(ques_list[i]+' '+ans_list[i])
    }

    $('.prod_cat span').text(prod_cat_list[index])
    $('.prod_idx span').text(prod_idx_list[index])

    var url = "https://www.amazon.com/dp/" + prod_idx_list[index]
    $("a").prop("href", url)
    
    $('input[type="radio"]').prop('checked', false);
    
    $(document).on('click', 'a', function(e){ 
        e.preventDefault(); 
        var url = $(this).attr('href'); 
        window.open(url, '_blank');
    });

    $('#submit_panel').hide(0).delay(1000).show(0);  

    $('#error_example').show();
    $('.error_btn').click(function(){

        if ($('.error_btn').text() == 'show examples'){
            $('.error_btn').text('hide examples');
            $('#error_example').show();
        }
        else if ($('.error_btn').text() == 'hide examples'){
            $('.error_btn').text('show examples');
            $('#error_example').hide();
        }
    }); 

    // real-time check answer 1
    $('#Textarea1').on( 'keyup', function () {

        var ori_sum = ques_list[0]+' '+ans_list[0]
        var edit_sum = $('#Textarea1').val()
        var error_cnt = 0

        // check submit
        if (edit_sum.replace(/\s/g, '') !== '' && click1 == -1){
            check_submit += 1
            click1 = 1
        }
        else if (edit_sum.replace(/\s/g, '') === '' && click1 == 1){
            check_submit -= 1
            click1 = -1
        }
        $('#submit_btn span').text(check_submit)

        // check LCS
        lcs_sim = lcs(ori_sum, edit_sum, ori_sum.length, edit_sum.length)/ori_sum.length
        // console.log(lcs_sim)

        if(lcs_sim > 0.7){
            $('#warn1_copy').addClass('warn_text');
            error_cnt += 1
        }
        else{
            $('#warn1_copy').removeClass('warn_text');
        }

        // check Yes/No 
        var yn_set = new Set(['Yes', 'yes', 'No', 'no'])
        var edit_sum_set = new Set(edit_sum.split(' '));
        var intersectionSet = edit_sum_set.intersection(yn_set);

        if (intersectionSet.size){
            $('#warn1_yes').addClass('warn_text');
            error_cnt += 1
        }
        else{
            $('#warn1_yes').removeClass('warn_text');
        }

        // check 'it'
        var it_term = new Set(["it","it's","its"])
        var first_word = new Set(edit_sum.trim().toLowerCase().split(/\b(\s)/));
        var head = first_word.values().next();
        if (it_term.has(head.value)){
            $('#warn1_self').addClass('warn_text');
            error_cnt += 1
        }
        else{
            $('#warn1_self').removeClass('warn_text');
        }

        //check contain question info
        var clean_ques = ques_list[0].replace(/[&\/\\#,+()$~%.'":*?<>{}]/g, '').split(/\b(\s)/);
        var intersectionSet = edit_sum_set.intersection(clean_ques);
        // console.log(clean_ques)
        // console.log(intersectionSet)

        if (intersectionSet.size) {
            $('#warn1_ignore').removeClass('warn_text');
        } else {
            $('#warn1_ignore').addClass('warn_text');
            error_cnt += 1
        }
        $('#QA_warn1 span').text(error_cnt)
        
    });

    // real-time check answer 2
    $('#Textarea2').on( 'keyup', function () {

        var ori_sum = ques_list[1]+' '+ans_list[1]
        var edit_sum = $('#Textarea2').val()
        var error_cnt = 0

        // check submit
        if (edit_sum.replace(/\s/g, '') !== '' && click2 == -1){
            check_submit += 1
            click2 = 1
        }
        else if (edit_sum.replace(/\s/g, '') === '' && click2 == 1){
            check_submit -= 1
            click2 = -1
        }
        $('#submit_btn span').text(check_submit)

        // check LCS
        lcs_sim = lcs(ori_sum, edit_sum, ori_sum.length, edit_sum.length)/ori_sum.length
        // console.log(lcs_sim)

        if(lcs_sim > 0.7){
            $('#warn2_copy').addClass('warn_text');
            error_cnt += 1
        }
        else{
            $('#warn2_copy').removeClass('warn_text');
        }

        // check Yes/No 
        var yn_set = new Set(['Yes', 'yes', 'No', 'no'])
        var edit_sum_set = new Set(edit_sum.split(' '));
        var intersectionSet = edit_sum_set.intersection(yn_set);

        if (intersectionSet.size){
            $('#warn2_yes').addClass('warn_text');
            error_cnt += 1
        }
        else{
            $('#warn2_yes').removeClass('warn_text');
        }

        // check 'it'
        var it_term = new Set(["it","it's","its"])
        var first_word = new Set(edit_sum.trim().toLowerCase().split(/\b(\s)/));
        var head = first_word.values().next();
        if (it_term.has(head.value)){
            $('#warn2_self').addClass('warn_text');
            error_cnt += 1
        }
        else{
            $('#warn2_self').removeClass('warn_text');
        }

        //check contain question info
        var clean_ques = ques_list[1].replace(/[&\/\\#,+()$~%.'":*?<>{}]/g, '').split(/\b(\s)/);
        var intersectionSet = edit_sum_set.intersection(clean_ques);
        // console.log(clean_ques)
        // console.log(intersectionSet)

        if (intersectionSet.size) {
            $('#warn2_ignore').removeClass('warn_text');
        } else {
            $('#warn2_ignore').addClass('warn_text');
            error_cnt += 1
        }
        $('#QA_warn2 span').text(error_cnt)
        
    });

    // real-time check answer 3
    $('#Textarea3').on( 'keyup', function () {

        var ori_sum = ques_list[2]+' '+ans_list[2]
        var edit_sum = $('#Textarea3').val()
        var error_cnt = 0

        // check submit
        if (edit_sum.replace(/\s/g, '') !== '' && click3 == -1){
            check_submit += 1
            click3 = 1
        }
        else if (edit_sum.replace(/\s/g, '') === '' && click3 == 1){
            check_submit -= 1
            click3 = -1
        }
        $('#submit_btn span').text(check_submit)

        // check LCS
        lcs_sim = lcs(ori_sum, edit_sum, ori_sum.length, edit_sum.length)/ori_sum.length
        // console.log(lcs_sim)

        if(lcs_sim > 0.7){
            $('#warn3_copy').addClass('warn_text');
            error_cnt += 1
        }
        else{
            $('#warn3_copy').removeClass('warn_text');
        }

        // check Yes/No 
        var yn_set = new Set(['Yes', 'yes', 'No', 'no'])
        var edit_sum_set = new Set(edit_sum.split(' '));
        var intersectionSet = edit_sum_set.intersection(yn_set);

        if (intersectionSet.size){
            $('#warn3_yes').addClass('warn_text');
            error_cnt += 1
        }
        else{
            $('#warn3_yes').removeClass('warn_text');
        }

        // check 'it'
        var it_term = new Set(["it","it's","its"])
        var first_word = new Set(edit_sum.trim().toLowerCase().split(/\b(\s)/));
        var head = first_word.values().next();
        if (it_term.has(head.value)){
            $('#warn3_self').addClass('warn_text');
            error_cnt += 1
        }
        else{
            $('#warn3_self').removeClass('warn_text');
        }

        //check contain question info
        var clean_ques = ques_list[2].replace(/[&\/\\#,+()$~%.'":*?<>{}]/g, '').split(/\b(\s)/);
        var intersectionSet = edit_sum_set.intersection(clean_ques);
        // console.log(clean_ques)
        // console.log(intersectionSet)

        if (intersectionSet.size) {
            $('#warn3_ignore').removeClass('warn_text');
        } else {
            $('#warn3_ignore').addClass('warn_text');
            error_cnt += 1
        }
        $('#QA_warn3 span').text(error_cnt)
        
    });

    // real-time check answer 4
    $('#Textarea4').on( 'keyup', function () {

        var ori_sum = ques_list[3]+' '+ans_list[3]
        var edit_sum = $('#Textarea4').val()
        var error_cnt = 0

        // check submit
        if (edit_sum.replace(/\s/g, '') !== '' && click4 == -1){
            check_submit += 1
            click4 = 1
        }
        else if (edit_sum.replace(/\s/g, '') === '' && click4 == 1){
            check_submit -= 1
            click4 = -1
        }
        $('#submit_btn span').text(check_submit)

        // check LCS
        lcs_sim = lcs(ori_sum, edit_sum, ori_sum.length, edit_sum.length)/ori_sum.length
        // console.log(lcs_sim)

        if(lcs_sim > 0.7){
            $('#warn4_copy').addClass('warn_text');
            error_cnt += 1
        }
        else{
            $('#warn4_copy').removeClass('warn_text');
        }

        // check Yes/No 
        var yn_set = new Set(['Yes', 'yes', 'No', 'no'])
        var edit_sum_set = new Set(edit_sum.split(' '));
        var intersectionSet = edit_sum_set.intersection(yn_set);

        if (intersectionSet.size){
            $('#warn4_yes').addClass('warn_text');
            error_cnt += 1
        }
        else{
            $('#warn4_yes').removeClass('warn_text');
        }

        // check 'it'
        var it_term = new Set(["it","it's","its"])
        var first_word = new Set(edit_sum.trim().toLowerCase().split(/\b(\s)/));
        var head = first_word.values().next();
        if (it_term.has(head.value)){
            $('#warn4_self').addClass('warn_text');
            error_cnt += 1
        }
        else{
            $('#warn4_self').removeClass('warn_text');
        }

        //check contain question info
        var clean_ques = ques_list[3].replace(/[&\/\\#,+()$~%.'":*?<>{}]/g, '').split(/\b(\s)/);
        var intersectionSet = edit_sum_set.intersection(clean_ques);
        // console.log(clean_ques)
        // console.log(intersectionSet)

        if (intersectionSet.size) {
            $('#warn4_ignore').removeClass('warn_text');
        } else {
            $('#warn4_ignore').addClass('warn_text');
            error_cnt += 1
        }
        $('#QA_warn4 span').text(error_cnt)
        
    });

    // real-time check answer 5
    $('#Textarea5').on( 'keyup', function () {

        var ori_sum = ques_list[4]+' '+ans_list[4]
        var edit_sum = $('#Textarea5').val()
        var error_cnt = 0

        // check submit
        if (edit_sum.replace(/\s/g, '') !== '' && click5 == -1){
            check_submit += 1
            click5 = 1
        }
        else if (edit_sum.replace(/\s/g, '') === '' && click5 == 1){
            check_submit -= 1
            click5 = -1
        }
        $('#submit_btn span').text(check_submit)

        // check LCS
        lcs_sim = lcs(ori_sum, edit_sum, ori_sum.length, edit_sum.length)/ori_sum.length
        // console.log(lcs_sim)

        if(lcs_sim > 0.7){
            $('#warn5_copy').addClass('warn_text');
            error_cnt += 1
        }
        else{
            $('#warn5_copy').removeClass('warn_text');
        }

        // check Yes/No 
        var yn_set = new Set(['Yes', 'yes', 'No', 'no'])
        var edit_sum_set = new Set(edit_sum.split(' '));
        var intersectionSet = edit_sum_set.intersection(yn_set);

        if (intersectionSet.size){
            $('#warn5_yes').addClass('warn_text');
            error_cnt += 1
        }
        else{
            $('#warn5_yes').removeClass('warn_text');
        }

        // check 'it'
        var it_term = new Set(["it","it's","its"])
        var first_word = new Set(edit_sum.trim().toLowerCase().split(/\b(\s)/));
        var head = first_word.values().next();
        if (it_term.has(head.value)){
            $('#warn5_self').addClass('warn_text');
            error_cnt += 1
        }
        else{
            $('#warn5_self').removeClass('warn_text');
        }

        //check contain question info
        var clean_ques = ques_list[4].replace(/[&\/\\#,+()$~%.'":*?<>{}]/g, '').split(/\b(\s)/);
        var intersectionSet = edit_sum_set.intersection(clean_ques);
        // console.log(clean_ques)
        // console.log(intersectionSet)

        if (intersectionSet.size) {
            $('#warn5_ignore').removeClass('warn_text');
        } else {
            $('#warn5_ignore').addClass('warn_text');
            error_cnt += 1
        }
        $('#QA_warn5 span').text(error_cnt)
        
    });

    // real-time check answer 6
    $('#Textarea6').on( 'keyup', function () {

        var ori_sum = ques_list[5]+' '+ans_list[5]
        var edit_sum = $('#Textarea6').val()
        var error_cnt = 0

        // check submit
        if (edit_sum.replace(/\s/g, '') !== '' && click6 == -1){
            check_submit += 1
            click6 = 1
        }
        else if (edit_sum.replace(/\s/g, '') === '' && click6 == 1){
            check_submit -= 1
            click6 = -1
        }
        $('#submit_btn span').text(check_submit)

        // check LCS
        lcs_sim = lcs(ori_sum, edit_sum, ori_sum.length, edit_sum.length)/ori_sum.length
        // console.log(lcs_sim)

        if(lcs_sim > 0.7){
            $('#warn6_copy').addClass('warn_text');
            error_cnt += 1
        }
        else{
            $('#warn6_copy').removeClass('warn_text');
        }

        // check Yes/No 
        var yn_set = new Set(['Yes', 'yes', 'No', 'no'])
        var edit_sum_set = new Set(edit_sum.split(' '));
        var intersectionSet = edit_sum_set.intersection(yn_set);

        if (intersectionSet.size){
            $('#warn6_yes').addClass('warn_text');
            error_cnt += 1
        }
        else{
            $('#warn6_yes').removeClass('warn_text');
        }

        // check 'it'
        var it_term = new Set(["it","it's","its"])
        var first_word = new Set(edit_sum.trim().toLowerCase().split(/\b(\s)/));
        var head = first_word.values().next();
        if (it_term.has(head.value)){
            $('#warn6_self').addClass('warn_text');
            error_cnt += 1
        }
        else{
            $('#warn6_self').removeClass('warn_text');
        }

        //check contain question info
        var clean_ques = ques_list[5].replace(/[&\/\\#,+()$~%.'":*?<>{}]/g, '').split(/\b(\s)/);
        var intersectionSet = edit_sum_set.intersection(clean_ques);
        // console.log(clean_ques)
        // console.log(intersectionSet)

        if (intersectionSet.size) {
            $('#warn6_ignore').removeClass('warn_text');
        } else {
            $('#warn6_ignore').addClass('warn_text');
            error_cnt += 1
        }
        $('#QA_warn6 span').text(error_cnt)
        
    });

    // real-time check answer 7
    $('#Textarea7').on( 'keyup', function () {

        var ori_sum = ques_list[6]+' '+ans_list[6]
        var edit_sum = $('#Textarea7').val()
        var error_cnt = 0

        // check submit
        if (edit_sum.replace(/\s/g, '') !== '' && click7 == -1){
            check_submit += 1
            click7 = 1
        }
        else if (edit_sum.replace(/\s/g, '') === '' && click7 == 1){
            check_submit -= 1
            click7 = -1
        }
        $('#submit_btn span').text(check_submit)

        // check LCS
        lcs_sim = lcs(ori_sum, edit_sum, ori_sum.length, edit_sum.length)/ori_sum.length
        // console.log(lcs_sim)

        if(lcs_sim > 0.7){
            $('#warn7_copy').addClass('warn_text');
            error_cnt += 1
        }
        else{
            $('#warn7_copy').removeClass('warn_text');
        }

        // check Yes/No 
        var yn_set = new Set(['Yes', 'yes', 'No', 'no'])
        var edit_sum_set = new Set(edit_sum.split(' '));
        var intersectionSet = edit_sum_set.intersection(yn_set);

        if (intersectionSet.size){
            $('#warn7_yes').addClass('warn_text');
            error_cnt += 1
        }
        else{
            $('#warn7_yes').removeClass('warn_text');
        }

        // check 'it'
        var it_term = new Set(["it","it's","its"])
        var first_word = new Set(edit_sum.trim().toLowerCase().split(/\b(\s)/));
        var head = first_word.values().next();
        if (it_term.has(head.value)){
            $('#warn7_self').addClass('warn_text');
            error_cnt += 1
        }
        else{
            $('#warn7_self').removeClass('warn_text');
        }

        //check contain question info
        var clean_ques = ques_list[6].replace(/[&\/\\#,+()$~%.'":*?<>{}]/g, '').split(/\b(\s)/);
        var intersectionSet = edit_sum_set.intersection(clean_ques);
        // console.log(clean_ques)
        // console.log(intersectionSet)

        if (intersectionSet.size) {
            $('#warn7_ignore').removeClass('warn_text');
        } else {
            $('#warn7_ignore').addClass('warn_text');
            error_cnt += 1
        }
        $('#QA_warn7 span').text(error_cnt)
        
    });

    // real-time check answer 8
    $('#Textarea8').on( 'keyup', function () {

        var ori_sum = ques_list[7]+' '+ans_list[7]
        var edit_sum = $('#Textarea8').val()
        var error_cnt = 0

        // check submit
        if (edit_sum.replace(/\s/g, '') !== '' && click8 == -1){
            check_submit += 1
            click8 = 1
        }
        else if (edit_sum.replace(/\s/g, '') === '' && click8 == 1){
            check_submit -= 1
            click8 = -1
        }
        $('#submit_btn span').text(check_submit)

        // check LCS
        lcs_sim = lcs(ori_sum, edit_sum, ori_sum.length, edit_sum.length)/ori_sum.length
        // console.log(lcs_sim)

        if(lcs_sim > 0.7){
            $('#warn8_copy').addClass('warn_text');
            error_cnt += 1
        }
        else{
            $('#warn8_copy').removeClass('warn_text');
        }

        // check Yes/No 
        var yn_set = new Set(['Yes', 'yes', 'No', 'no'])
        var edit_sum_set = new Set(edit_sum.split(' '));
        var intersectionSet = edit_sum_set.intersection(yn_set);

        if (intersectionSet.size){
            $('#warn8_yes').addClass('warn_text');
            error_cnt += 1
        }
        else{
            $('#warn8_yes').removeClass('warn_text');
        }

        // check 'it'
        var it_term = new Set(["it","it's","its"])
        var first_word = new Set(edit_sum.trim().toLowerCase().split(/\b(\s)/));
        var head = first_word.values().next();
        if (it_term.has(head.value)){
            $('#warn8_self').addClass('warn_text');
            error_cnt += 1
        }
        else{
            $('#warn8_self').removeClass('warn_text');
        }

        //check contain question info
        var clean_ques = ques_list[7].replace(/[&\/\\#,+()$~%.'":*?<>{}]/g, '').split(/\b(\s)/);
        var intersectionSet = edit_sum_set.intersection(clean_ques);
        // console.log(clean_ques)
        // console.log(intersectionSet)

        if (intersectionSet.size) {
            $('#warn8_ignore').removeClass('warn_text');
        } else {
            $('#warn8_ignore').addClass('warn_text');
            error_cnt += 1
        }
        $('#QA_warn8 span').text(error_cnt)
        
    });
 
    // time start
    start()

    $(document).on('click', '#submit_btn', function(evt) {

        setTimeout(function(){
        $("#warning").hide();
        },2000)

        if (check_submit != 8){
            $("#warning").show();
            var warm_txt = 'Please summarize all Q-A pairs !!!'
            $("#warning").text(warm_txt);
            return;
        }

        for (i = 0; i < 8; i++){
            var ori_sum = ques_list[i]+' '+ans_list[i]
            var edit_sum = $('#Textarea'+parseInt(i+1)).val()
            // check LCS
            lcs_sim = lcs(ori_sum, edit_sum, ori_sum.length, edit_sum.length)/ori_sum.length
            if(lcs_sim > 0.8 || lcs_sim < 0.2){
                $("#warning").show();
                var warm_txt = 'Please edit Q'+parseInt(i+1)+' & A'+parseInt(i+1)+' summary moderately !!!'
                $("#warning").text(warm_txt);
                return;
            } 
            // check Yes/No 
            var yn_set = new Set(['Yes', 'No'])
            var edit_sum_set = new Set(edit_sum.split(' '));
            var intersectionSet = edit_sum_set.intersection(yn_set);

            if (intersectionSet.size){
                $("#warning").show();
                var warm_txt = 'Do not use Yes/No in Summary'+parseInt(i+1)+' !!!'
                $("#warning").text(warm_txt);
                return;
            }

            //check Q&A format
            var q_term = new Set(['who', 'which', 'when', 'where', 'why', 'how', 'what','am', 'are', 'is', 'was', 'were',
                                  'have', 'has', 'had', 'shall', 'will', 'should', 'would', 'do', 'does', 'did'])
            var first_word = new Set(edit_sum.trim().toLowerCase().split(' '));
            var head = first_word.values().next();
            if (q_term.has(head.value) || edit_sum.includes('?')){
                $("#warning").show();
                var warm_txt = 'Do not use Q&A format in Summary '+parseInt(i+1)+' !!!'
                $("#warning").text(warm_txt);
                return;
            }

            // check 'it'
            var it_term = new Set(["it","it's","its"])
            var first_word = new Set(edit_sum.trim().toLowerCase().split(' '));
            var head = first_word.values().next();
            if (it_term.has(head.value)){
                $("#warning").show();
                var warm_txt = 'Refer the product name instead of "It" in Summary '+parseInt(i+1)+' !!!'
                $("#warning").text(warm_txt);
                return;
            }

            // check first person pronouns
            var first_prons_set = new Set(['I', 'Me', 'me', 'My', 'my', 'Mine', 'mine', 'Our', 'our', 'Ours', 'ours', 'We', 'we'])
            var intersectionSet = edit_sum_set.intersection(first_prons_set);
            
            if (intersectionSet.size){
                $("#warning").show();
                var warm_txt = 'Do not use first person pronouns in Summary '+parseInt(i+1)+' !!!'
                $("#warning").text(warm_txt);
                return;
            }
        }

        $('#submit_panel').hide(0).delay(1000).show(0);
        var ques_txt = []
        var ans_txt = []
        var qa_summary = []
        var elapse_time = 0
        for (i = 0; i < 8; i++){
            ques_txt.push(ques_list[i]);
            ans_txt.push(ans_list[i]); 
            var textarea = $('#Textarea'+parseInt(i+1)).val()
            qa_summary.push(textarea)
        }

        // time end
        elapse_time = end()

        index += 8
        
        // save data information before next figure
        var prod_cat = $('.prod_cat span').text()
        var prod_idx = $('.prod_idx span').text()
        var data_info = {'prod_cat':prod_cat, 'prod_idx':prod_idx,
                        'ques':ques_txt, 'ans':ans_txt, 'qa_summary':qa_summary,
                        'time':elapse_time}

        qa_info_list.push(data_info)
        //localStorage.setItem("data_info", JSON.stringify(qa_info_list));
        //localStorage.setItem("progress_index", JSON.stringify(index));

        if (index == len) { 
            $('#question_form').hide();
            $("#answer").val(JSON.stringify(qa_info_list));
            console.log(qa_info_list)
            // $("#mturk_form").submit();
        }

        // update data info
        // $('.prod_cat span').text(prod_cat_list[index])
        // $('.prod_idx span').text(prod_idx_list[index])
        // for (i = index; i < index+8; i++){
        //     var q_txt = '.ques_txt_' + parseInt(i%8+1) + ' span'
        //     var a_txt = '.ans_txt_' + parseInt(i%8+1) + ' span'
        //     var qa_sum = '#Textarea' + parseInt(i%8+1)
        //     $(q_txt).text(ques_list[i])
        //     $(a_txt).text(ans_list[i])
        //     $(qa_sum).val(ques_list[i]+' '+ans_list[i])
        // }

        // $('input[type="radio"]').prop('checked', false);

        // $('.progress-bar').attr('aria-valuenow',index);
        // $('.progress-bar').attr('style','width:'+(index/len*100).toFixed(2)+'%');
        // $('.progress-bar').text((index/len*100).toFixed(2)+'%');
        // $('#prog_count').text((index/len*100).toFixed(2)+'% '+'complete');

        
        // var url = "https://www.amazon.com/dp/" + prod_idx_list[index]
        // $("a").prop("href", url)

        // start()
    });
});

var temp;
