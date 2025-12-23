import 'package:flutter/material.dart';

void main() {
  runApp(const MyProfileApp());
}

class MyProfileApp extends StatelessWidget {
  const MyProfileApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: Scaffold(
        backgroundColor:
            Colors.teal, // පසුබිම් පාට ( කැමති එකක් දාගන්න පුලුවන්)
        body: SafeArea(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center, // මැදට ගන්න
            children: [
              // 1. Profile Picture (රවුම් ෆොටෝ එක)
              const CircleAvatar(
                radius: 60.0, // රවුමේ ප්‍රමාණය
                backgroundImage: AssetImage('assets/me.png'),
              ),

              // 2. Name (නම)
              const Text(
                'Dinesh Madusanka',
                style: TextStyle(
                  fontSize: 35.0,
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                  fontFamily: 'Arial', // පස්සේ අපි ලස්සන Font එකක් දාමු
                ),
              ),

              // 3. Job Title (තනතුර)
              Text(
                'FOUNDER - INFLIXT GLOBAL',
                style: TextStyle(
                  fontSize: 18.0,
                  color: Colors.teal.shade100,
                  letterSpacing: 2.5, // අකුරු අතර පරතරය
                  fontWeight: FontWeight.bold,
                ),
              ),

              // ඉරක් ගහන්න (Divider)
              SizedBox(
                height: 20.0,
                width: 150.0,
                child: Divider(color: Colors.teal.shade100),
              ),

              // 4. Phone Number Card
              Card(
                margin: const EdgeInsets.symmetric(
                  vertical: 10.0,
                  horizontal: 25.0,
                ),
                child: ListTile(
                  leading: const Icon(Icons.phone, color: Colors.teal),
                  title: Text(
                    '+94 77 407 0056',
                    style: TextStyle(
                      color: Colors.teal.shade900,
                      fontSize: 20.0,
                    ),
                  ),
                ),
              ),

              // 5. Email Card
              Card(
                margin: const EdgeInsets.symmetric(
                  vertical: 10.0,
                  horizontal: 25.0,
                ),
                child: ListTile(
                  leading: const Icon(Icons.email, color: Colors.teal),
                  title: Text(
                    'hello@dineshmadusanka.dev',
                    style: TextStyle(
                      color: Colors.teal.shade900,
                      fontSize: 18.0, // ඊමේල් එක දිග වැඩි නම් මේක 16.0 කරන්න
                    ),
                  ),
                ),
              ),

              // 6. Website Card (අමතරව දැම්මා)
              Card(
                margin: const EdgeInsets.symmetric(
                  vertical: 10.0,
                  horizontal: 25.0,
                ),
                child: ListTile(
                  leading: const Icon(Icons.language, color: Colors.teal),
                  title: Text(
                    'www.inflixtglobal.com',
                    style: TextStyle(
                      color: Colors.teal.shade900,
                      fontSize: 18.0,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
