package com.example.leafdiseasedetection

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import com.google.firebase.database.FirebaseDatabase

class MainActivity : AppCompatActivity() {
     lateinit var name: EditText
    lateinit var phone: EditText
    lateinit var btncon: Button
       override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
           name=findViewById(R.id.editName)
           phone=findViewById(R.id.editPhone)
          btncon=findViewById<Button>(R.id.btnContinue)
        btncon.setOnClickListener{
            saveData()



        }
    }
    public fun saveData()
    {
        val nametxt=name.text.toString()
        val phonetxt=phone.text.toString()
        if(nametxt.isEmpty()||phonetxt.isEmpty()||phonetxt.length!=10) {
            Toast.makeText(applicationContext, "Enter valid data", Toast.LENGTH_LONG).show()
        return
        }
        val intent= Intent(this,DetectionActivity::class.java)
            intent.putExtra("Name",nametxt)
            intent.putExtra("Phone",phonetxt)
           //name.setText(null)
          // phone.setText(null)

            startActivity(intent)
    }
}
